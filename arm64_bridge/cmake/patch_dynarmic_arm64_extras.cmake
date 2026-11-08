if(NOT DEFINED DYNARMIC_SOURCE_DIR)
    message(FATAL_ERROR "DYNARMIC_SOURCE_DIR is required")
endif()

function(require_replace file_path old_text new_text description)
    file(READ "${file_path}" file_content)
    string(FIND "${file_content}" "${old_text}" anchor_position)
    if(anchor_position EQUAL -1)
        message(FATAL_ERROR "Pinned Dynarmic source no longer matches ${description}: ${file_path}")
    endif()
    string(REPLACE "${old_text}" "${new_text}" file_content "${file_content}")
    file(WRITE "${file_path}" "${file_content}")
endfunction()

# The A32 frontend in the pinned revision leaves VADDHN/VRADDHN undecoded.
# Unity's audio mixer executes VRADDHN, so implement both architectural forms.
set(decoder "${DYNARMIC_SOURCE_DIR}/src/dynarmic/frontend/A32/decoder/asimd.inc")
file(READ "${decoder}" decoder_content)
string(FIND "${decoder_content}" [=[111100111Dzznnnndddd0100N0M0mmmm]=] vaddhn_patched)
if(vaddhn_patched EQUAL -1)
    require_replace("${decoder}"
        [=[//INST(asimd_VADDHN,          "VADDHN",                   "111100101-----------0100-0-0----") // ASIMD
//INST(asimd_VRADDHN,         "VRADDHN",                  "111100111-----------0100-0-0----") // ASIMD]=]
        [=[INST(asimd_VADDHN,          "VADDHN",                   "111100101Dzznnnndddd0100N0M0mmmm") // ASIMD
INST(asimd_VRADDHN,         "VRADDHN",                  "111100111Dzznnnndddd0100N0M0mmmm") // ASIMD]=]
        "the A32 VADDHN decoder anchors")

    set(translator_header "${DYNARMIC_SOURCE_DIR}/src/dynarmic/frontend/A32/translate/impl/a32_translate_impl.h")
    require_replace("${translator_header}"
        [=[    bool asimd_VPMIN_float(bool D, bool sz, size_t Vn, size_t Vd, bool N, bool Q, bool M, size_t Vm);
    bool asimd_VRECPS(bool D, bool sz, size_t Vn, size_t Vd, bool N, bool Q, bool M, size_t Vm);]=]
        [=[    bool asimd_VPMIN_float(bool D, bool sz, size_t Vn, size_t Vd, bool N, bool Q, bool M, size_t Vm);
    bool asimd_VADDHN(bool D, size_t sz, size_t Vn, size_t Vd, bool N, bool M, size_t Vm);
    bool asimd_VRADDHN(bool D, size_t sz, size_t Vn, size_t Vd, bool N, bool M, size_t Vm);
    bool asimd_VRECPS(bool D, bool sz, size_t Vn, size_t Vd, bool N, bool Q, bool M, size_t Vm);]=]
        "the A32 VADDHN declarations")

    set(translator_impl "${DYNARMIC_SOURCE_DIR}/src/dynarmic/frontend/A32/translate/impl/asimd_three_regs.cpp")
    require_replace("${translator_impl}"
        [=[bool TranslatorVisitor::asimd_VRECPS(bool D, bool sz, size_t Vn, size_t Vd, bool N, bool Q, bool M, size_t Vm) {]=]
        [=[// BRIDGE_A32_VADDHN: operations required by Unity's ARMv7 audio mixer.
bool TranslatorVisitor::asimd_VRADDHN(bool D, size_t sz, size_t Vn, size_t Vd, bool N, bool M, size_t Vm) {
    if (sz == 0b11) {
        return UndefinedInstruction();
    }

    const size_t esize = 8U << sz;
    const auto d = ToVector(false, Vd, D);
    const auto n = ToVector(true, Vn, N);
    const auto m = ToVector(true, Vm, M);

    static constexpr std::array<u64, 3> rounding_terms{0x0080008000800080ull,
                                                       0x0000800000008000ull,
                                                       0x8000000080000000ull};
    const auto reg_n = ir.GetVector(n);
    const auto reg_m = ir.GetVector(m);
    const auto wide_sum = ir.VectorAdd(esize * 2, reg_n, reg_m);
    const auto rounding_term = ir.Imm64(rounding_terms[sz]);
    const auto rounded = ir.VectorAdd(esize * 2, wide_sum, ir.VectorBroadcast(64, rounding_term));
    const auto result = ir.VectorNarrow(esize * 2, rounded);

    ir.SetVector(d, result);
    return true;
}

bool TranslatorVisitor::asimd_VADDHN(bool D, size_t sz, size_t Vn, size_t Vd, bool N, bool M, size_t Vm) {
    if (sz == 0b11) {
        return UndefinedInstruction();
    }

    const size_t esize = 8U << sz;
    const auto d = ToVector(false, Vd, D);
    const auto n = ToVector(true, Vn, N);
    const auto m = ToVector(true, Vm, M);

    const auto reg_n = ir.GetVector(n);
    const auto reg_m = ir.GetVector(m);
    const auto wide_sum = ir.VectorAdd(esize * 2, reg_n, reg_m);
    const auto result = ir.VectorNarrow(esize * 2, wide_sum);

    ir.SetVector(d, result);
    return true;
}

bool TranslatorVisitor::asimd_VRECPS(bool D, bool sz, size_t Vn, size_t Vd, bool N, bool Q, bool M, size_t Vm) {]=]
        "the A32 VADDHN implementation anchor")
    message(STATUS "Applied A32 VADDHN/VRADDHN patch to pinned Dynarmic")
endif()

# Dynarmic exposes fastmem_exclusive_access, but the pinned ARM64 backend does
# not consume it. Emit host LDAXR/STLXR directly for the bridge's complete,
# silently-mirrored 32-bit guest address space.
set(emit_header "${DYNARMIC_SOURCE_DIR}/src/dynarmic/backend/arm64/emit_arm64.h")
file(READ "${emit_header}" emit_header_content)
string(FIND "${emit_header_content}" "bool fastmem_exclusive_access;" fastmem_patched)
if(fastmem_patched EQUAL -1)
    require_replace("${emit_header}"
        [=[    bool silently_mirror_fastmem;

    // Timing]=]
        [=[    bool silently_mirror_fastmem;
    bool fastmem_exclusive_access;

    // Timing]=]
        "the ARM64 fastmem EmitConfig field")

    foreach(address_space IN ITEMS a32_address_space.cpp a64_address_space.cpp)
        set(address_path "${DYNARMIC_SOURCE_DIR}/src/dynarmic/backend/arm64/${address_space}")
        file(READ "${address_path}" address_content)
        string(REPLACE
            [=[        .silently_mirror_fastmem = true,

        .wall_clock_cntpct]=]
            [=[        .silently_mirror_fastmem = true,
        .fastmem_exclusive_access = conf.fastmem_exclusive_access,

        .wall_clock_cntpct]=]
            address_content "${address_content}")
        string(REPLACE
            [=[        .silently_mirror_fastmem = conf.silently_mirror_fastmem,

        .wall_clock_cntpct]=]
            [=[        .silently_mirror_fastmem = conf.silently_mirror_fastmem,
        .fastmem_exclusive_access = conf.fastmem_exclusive_access,

        .wall_clock_cntpct]=]
            address_content "${address_content}")
        file(WRITE "${address_path}" "${address_content}")
    endforeach()

    set(a32_memory "${DYNARMIC_SOURCE_DIR}/src/dynarmic/backend/arm64/emit_arm64_a32_memory.cpp")
    require_replace("${a32_memory}"
        [=[void EmitIR<IR::Opcode::A32ClearExclusive>(oaknut::CodeGenerator& code, EmitContext&, IR::Inst*) {
    code.STR(WZR, Xstate, offsetof(A32JitState, exclusive_state));]=]
        [=[void EmitIR<IR::Opcode::A32ClearExclusive>(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst*) {
    if (ctx.conf.fastmem_exclusive_access) {
        code.CLREX();
    }
    code.STR(WZR, Xstate, offsetof(A32JitState, exclusive_state));]=]
        "the A32 CLREX emitter")

    set(memory "${DYNARMIC_SOURCE_DIR}/src/dynarmic/backend/arm64/emit_arm64_memory.cpp")
    require_replace("${memory}"
        [=[}  // namespace

template<size_t bitsize>
void EmitReadMemory]=]
        [=[template<size_t bitsize>
void FastmemEmitExclusiveReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(bitsize == 8 || bitsize == 16 || bitsize == 32 || bitsize == 64);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = ctx.reg_alloc.WriteReg<std::max<std::size_t>(bitsize, 32)>(inst);
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue);

    // BRIDGE_ARM64_FASTMEM_EXCLUSIVE: use the host PE's local monitor for the
    // directly-mapped 32-bit guest address space.
    code.MOV(Wscratch0, 1);
    code.STRB(Wscratch0, Xstate, ctx.conf.state_exclusive_state_offset);
    code.ADD(Xscratch0, Xfastmem, oaknut::RReg{Xaddr->toW()}, oaknut::AddSubExt::UXTW);
    if constexpr (bitsize == 8) {
        code.LDAXRB(oaknut::WReg{Rvalue->index()}, Xscratch0);
    } else if constexpr (bitsize == 16) {
        code.LDAXRH(oaknut::WReg{Rvalue->index()}, Xscratch0);
    } else if constexpr (bitsize == 32) {
        code.LDAXR(oaknut::WReg{Rvalue->index()}, Xscratch0);
    } else {
        code.LDAXR(oaknut::XReg{Rvalue->index()}, Xscratch0);
    }
}

template<size_t bitsize>
void FastmemEmitExclusiveWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    static_assert(bitsize == 8 || bitsize == 16 || bitsize == 32 || bitsize == 64);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    auto Xaddr = ctx.reg_alloc.ReadX(args[1]);
    auto Rvalue = ctx.reg_alloc.ReadReg<std::max<std::size_t>(bitsize, 32)>(args[2]);
    auto Wstatus = ctx.reg_alloc.WriteW(inst);
    ctx.fpsr.Spill();
    ctx.reg_alloc.SpillFlags();
    RegAlloc::Realize(Xaddr, Rvalue, Wstatus);

    oaknut::Label end;
    code.MOV(Wstatus, 1);
    code.LDRB(Wscratch0, Xstate, ctx.conf.state_exclusive_state_offset);
    code.CBZ(Wscratch0, end);
    code.STRB(WZR, Xstate, ctx.conf.state_exclusive_state_offset);
    code.ADD(Xscratch0, Xfastmem, oaknut::RReg{Xaddr->toW()}, oaknut::AddSubExt::UXTW);
    if constexpr (bitsize == 8) {
        code.STLXRB(Wstatus, Rvalue->toW(), Xscratch0);
    } else if constexpr (bitsize == 16) {
        code.STLXRH(Wstatus, Rvalue->toW(), Xscratch0);
    } else if constexpr (bitsize == 32) {
        code.STLXR(Wstatus, Rvalue->toW(), Xscratch0);
    } else {
        code.STLXR(Wstatus, Rvalue->toX(), Xscratch0);
    }
    code.l(end);
}

}  // namespace

template<size_t bitsize>
void EmitReadMemory]=]
        "the ARM64 exclusive fastmem helpers")

    require_replace("${memory}"
        [=[void EmitExclusiveReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    CallbackOnlyEmitExclusiveReadMemory<bitsize>(code, ctx, inst);
}]=]
        [=[void EmitExclusiveReadMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if constexpr (bitsize <= 64) {
        if (ctx.conf.fastmem_exclusive_access && ctx.conf.fastmem_pointer &&
            ctx.conf.fastmem_address_space_bits == 32 && ctx.conf.silently_mirror_fastmem) {
            FastmemEmitExclusiveReadMemory<bitsize>(code, ctx, inst);
            return;
        }
    }
    CallbackOnlyEmitExclusiveReadMemory<bitsize>(code, ctx, inst);
}]=]
        "the ARM64 exclusive-read dispatch")

    require_replace("${memory}"
        [=[void EmitExclusiveWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    CallbackOnlyEmitExclusiveWriteMemory<bitsize>(code, ctx, inst);
}]=]
        [=[void EmitExclusiveWriteMemory(oaknut::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst) {
    if constexpr (bitsize <= 64) {
        if (ctx.conf.fastmem_exclusive_access && ctx.conf.fastmem_pointer &&
            ctx.conf.fastmem_address_space_bits == 32 && ctx.conf.silently_mirror_fastmem) {
            FastmemEmitExclusiveWriteMemory<bitsize>(code, ctx, inst);
            return;
        }
    }
    CallbackOnlyEmitExclusiveWriteMemory<bitsize>(code, ctx, inst);
}]=]
        "the ARM64 exclusive-write dispatch")
    message(STATUS "Applied ARM64 fastmem-exclusive patch to pinned Dynarmic")
endif()
