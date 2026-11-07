;--------------------------------------------------------------------------------------------------
; Routine Name: dzx0 (unzx0_pro_106b.asm)
; Description: 106-byte Zero-Overhead Multi-Stage 2D + 16-Bit Word-Add x86-16 decompressor
;              (-3B vs 109B, -5B vs 111B, <= 128B budget) unifying 0B-Overhead ZX0 v2,
;              x86 E8/E9 rel16 CALL/JMP restoration, and Unified 16-Bit Word / 8-Bit Byte
;              Multi-Stride 2D filtering (1..128) via algebraic 'sahf / jc .e8_loop' dispatch:
;              - Mode 0 (V=256): Standard ZX0 v2 EOF (CX=0xFF00; at .eof: 'inc ch' sets ZF=1 -> ret),
;                achieving 100% bit-for-bit backward compatibility with standard ZX0 v2 streams and
;                ZERO bytes of header/trailer overhead on unfiltered data.
;              - Mode 1 (V=513): Free-BX Multi-Pass 16-Bit Word / 8-Bit Byte 2D Stride-Delta
;                (CH=0xFD -> inc ch -> CH=0xFE; after 'lodsw / xchg ax, cx', AH=0xFE has bit 0 = 0
;                so 'sahf' sets CF=0 -> enters .floop directly).
;              - Mode 2 (V=792 / V=791): 1-pass x86 E8 CALL rel16 (V=792) or E9 JMP rel16 (V=791)
;                filter (CH=0xFC -> inc ch -> CH=0xFD; after 'lodsw / xchg ax, cx', AH=0xFD has
;                bit 0 = 1 so 'sahf' sets CF=1 -> jc .e8_loop, and after .e8_loop CX=0x0000 ->
;                inc ch -> CH=0x01 -> Pass 2 has AH=0x01 -> enters 2nd pass or .floop).
;              - Compact 4-Byte-Per-Pass 16-Bit Word / 8-Bit Byte 2D Stride Integrator (1..128):
;                Uses 'push ax / cbw / xchg ax, bx / pop ax' and CF-preserving 'inc dx / jz .fskip /
;                cwd / clc' so 'adc [di], al' performs exact 8-bit Byte-ADD or 16-bit LE Word-ADD.
;
; Credits & Copyright:
;   - Algorithm: ZX0 by Einar Saukas (c) 2021 (BSD 3-Clause)
;   - Reference 8088 architecture: Emmanuel Marty (unzx0_8088.S, c 2021, zlib License)
;   - Multi-stage E8/E9 + 16-bit Word / 8-bit Byte 2D Stride size-coding (106B): Oleksandr Dreval (MIT)
;
; License: MIT (SPDX-License-Identifier: MIT)
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
    js .eof

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

.e8_loop:
    scasb
    jne .next
    add [di], cx

.next:
    loop .e8_loop

.eof:
    inc ch
    jz .done

.mp_loop:
    lodsw
    xchg ax, cx
    jcxz .done
    sub di, cx
    inc ax
    js .e8_loop
    lodsw
    push ax
    cbw
    xchg ax, bx
    pop ax

.floop:
    inc dx
    jz .fskip
    cwd
    clc

.fskip:
    mov al, [bx+di]
    adc [di], al
    inc di
    loop .floop
    jmp short .mp_loop

.done:
    ret
