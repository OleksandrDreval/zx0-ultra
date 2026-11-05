;--------------------------------------------------------------------------------------------------
; Routine Name: dzx0 (unzx0_ultra_128b.asm)
; Description: 128-byte Flagship Multi-Modal 2D x86-16 decompressor (exact 128B budget)
;              unifying Asymmetric Split-Domain Variable-Bit Literals, Chained x86 E8/E9 rel16
;              CALL/JMP restoration, and Unified 16-Bit Word / 8-Bit Byte Multi-Pass 2D Stride-Delta:
;              - Asymmetric Split-Domain Variable-Bit Literals (3..9 bits/literal): at entry, reads a
;                16-bit header word into DX initializing DL with the first 7 bitstream bits +
;                sentinel bit 0 ('1'), and DH with literal sentinel 1..127. Inside .lit_loop,
;                'mov al, dh / call .get_bit / jnc .lit_bit / and al, 5 / inc ax' selects AL = DH
;                on CF=0 (encoding any hot subrange in 3..8 bits) and AL = (DH & 5) + 1 on CF=1
;                (selecting full 8-bit fallback [0..255] when (DH & 5)==0, 7-bit ASCII fallback
;                [0..127] when (DH & 5)==1, or 6-bit fallback [64..127] when (DH & 5)==4).
;              - Algebraic Multi-Mode EOF Dispatch & Fall-Through E8/E9 Post-Filter:
;                - Mode 0 (V=256 / V=257): Enters .mp_loop directly in .floop (for V=257, AH=0xFE
;                  so 'sahf' sets CF=0; for V=256 unfiltered, 0x0000 count exits via 'jcxz .done').
;                - Mode 1 (V=536 / V=535): For V=536 (CX=0xFDE8) or V=535 (CX=0xFDE9), AH=0xFD has
;                  bit 0 = 1 so 'sahf' sets CF=1 ('jc .e8_loop'), running 1 pass of .e8_loop with
;                  AL=0xE8 (CALL rel16) or AL=0xE9 (JMP rel16), leaving CX=0x0000 so Pass 2+ has
;                  AH=0x00 (CF=0) and transitions automatically to .floop.
;              - Compact 4-Byte-Per-Pass 16-Bit Word / 8-Bit Byte 2D Stride Integrator (1..128):
;                Reads count_u16 + packed mode_stride_u16 (AL = 256 - stride, AH = 0x80 for 16-bit
;                Word-mode or 0x00 for 8-bit Byte-mode), sets DX = 0x0000 (8-bit) or 0xFFFF (16-bit)
;                via 'cwd', and sign-extends AL via 'cbw / xchg ax, bx' (BH = 0xFF guaranteed!)
;                with zero constraints on the final match offset BX prior to EOF and full 64KB
;                (up to 65,535 bytes) count support via 'inc dx / jnz .fnext / ... / dec dx'.
;
; Credits & Copyright:
;   - Algorithm: ZX0 by Einar Saukas (c) 2021 (BSD 3-Clause)
;   - Reference 8088 architecture: Emmanuel Marty (unzx0_8088.S, c 2021, zlib License)
;   - Flagship 128B Multi-Modal 2D + Asymmetric Split-Domain size-coding: Oleksandr Dreval (MIT)
;
; License: MIT (SPDX-License-Identifier: MIT)
;--------------------------------------------------------------------------------------------------

bits 16

dzx0:
    xchg ax, cx
    lodsw
    xchg ax, dx

.literals:
    call .get_elias

.lit_loop:
    mov al, dh
    call .get_bit
    jnc .lit_bit
    and al, 5
    inc ax

.lit_bit:
    call .get_bit
    adc al, al
    jnc .lit_bit
    dec ax
    stosb
    loop .lit_loop
    call .get_bit
    jc .get_offset
    call .get_elias

.do_copy:
    push si
    lea si, [di+bx]
    rep movsb
    pop si
    call .get_bit
    jnc .literals

.get_offset:
    mov cl, 0xfe
    call .elias_loop
    inc cx
    js .eof

    mov ah, cl
    mov cx, 1
    lodsb
    xchg ax, bx
    rcr bx, 1
    call .got_bit
    inc cx
    jmp short .do_copy

.get_elias:
    inc cx

.elias_loop:
    call .get_bit

.got_bit:
    jc .done
    call .get_bit
    adc cx, cx
    jmp short .elias_loop

.get_bit:
    add dl, dl
    jnz .done
    mov dl, [si]
    inc si
    adc dl, dl

.done:
    ret

.e8_loop:
    scasb
    jne .next
    add [di], cx

.next:
    loop .e8_loop

.eof:
.mp_loop:
    lodsw
    xchg ax, cx
    jcxz .done
    sub di, cx
    sahf
    jc .e8_loop
    lodsw
    cwd
    cbw
    xchg ax, bx

.floop:
    mov al, [bx+di]
    add [di], al
    inc di
    inc dx
    jnz .fnext
    adc [di], dh
    dec dx
    dec dx

.fnext:
    loop .floop
    jmp short .mp_loop
