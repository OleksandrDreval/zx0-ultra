;--------------------------------------------------------------------------------------------------
; Routine Name: dzx0 (unzx0_67b.asm)
; Description: Ultra-compact 67-byte ZX0 v2 decompressor for 16-bit MS-DOS .COM executables
;              (DS = ES = SS = CS). 100% bit-for-bit compatible with standard ZX0 v2 streams.
;              Size: 67 bytes (-3 bytes vs 70-byte unzx0_x86.asm, -14 bytes vs 81-byte unzx0_8088.S).
;
; Credits & Copyright:
;   - Algorithm: ZX0 by Einar Saukas (c) 2021 (BSD 3-Clause)
;   - Reference 8088 architecture: Emmanuel Marty (unzx0_8088.S, c 2021, zlib License)
;   - Implementation & size-coding (67B): Oleksandr Dreval (MIT License)
;
; License: MIT (SPDX-License-Identifier: MIT)
;
; Key Micro-Optimizations over 70-byte unzx0_x86.asm:
;   [1] Replaced 'rcr bx, 1 / jc .got_offs / call .elias_bt / .got_offs:' (7 bytes) with
;       'rcr bx, 1 / call .got_bit' (5 bytes) = -2 bytes.
;       Because near CALL preserves CF, calling .got_bit directly executes 'jc .done' (ret) when
;       the length backtrack bit is 1, or falls into .elias_bt when 0.
;   [2] Replaced 'inc cl / jz DZX0_EOF_TARGET' (4 bytes) with 'inc cx / js DZX0_EOF_TARGET' (3 bytes)
;       = -1 byte.
;       On entry to .get_offset, CX is architecturally guaranteed 0x0000 after 'rep movsb'.
;       Seeding 'mov cl, 0xfe' sets CX = 0x00FE. Valid offsets (V = 1..255, 0..7 shifts) produce
;       CX in 0x00FE..0x7F7F, so 'inc cx' yields 0x00FF..0x7F80 (SF = 0, exact CL = -V, CF = 1).
;       The EOF marker (V = 256, 8 shifts) produces CX = 0xFEFF -> 'inc cx' yields 0xFF00 (SF = 1).
;--------------------------------------------------------------------------------------------------

bits 16

dzx0:
    mov al, 0x80
    dec bx

.literals:
    call .get_elias
    rep movsb
    add al, al
    jc .get_offset
    call .get_elias

.do_copy:
    push si
    lea si, [di+bx]
    rep movsb
    pop si
    add al, al
    jnc .literals

.get_offset:
    mov cl, 0xfe
    call .elias_loop
    inc cx
%ifndef DZX0_EOF_TARGET
%define DZX0_EOF_TARGET .done
%endif
    js DZX0_EOF_TARGET

    mov bh, cl
    mov cx, 1
    mov bl, [si]
    inc si
    rcr bx, 1
    call .got_bit
    inc cx
    jmp short .do_copy

.get_elias:
    mov cl, 1

.elias_loop:
    add al, al
    jnz .got_bit
    lodsb
    adc al, al

.got_bit:
    jc .done

.elias_bt:
    add al, al
    adc cx, cx
    jmp short .elias_loop

.done:
    ret
