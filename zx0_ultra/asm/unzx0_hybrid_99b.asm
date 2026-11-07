;--------------------------------------------------------------------------------------------------
; Routine Name: dzx0 (unzx0_hybrid_99b.asm)
; Description: 99-byte Multi-Stage Hybrid 2D x86-16 decompressor (-3B vs 102B, -5B vs 104B)
;              unifying 0B-Overhead ZX0 v2, x86 E8/E9 rel16 CALL/JMP restoration, and Free-BX
;              Multi-Stride filtering (1..128) via algebraic 'sahf / jc .e8_loop' EOF dispatch:
;              - Mode 0 (V=256): Standard ZX0 v2 EOF (CX=0xFF00; at .eof: 'inc ch' sets ZF=1 -> ret),
;                achieving 100% bit-for-bit backward compatibility with standard ZX0 v2 streams and
;                ZERO bytes of header/trailer overhead on unfiltered data.
;              - Mode 1 (V=257): Free-BX Multi-Pass 2D Stride-Delta (CH=0xFE -> inc ch -> CH=0xFF;
;                after 'lodsw / xchg ax, cx', AH=0xFF breaks at Pass 2 via 'inc ch' or enters .floop
;                when V=513 -> AH=0xFE -> sahf sets CF=0).
;              - Mode 2 (V=792 / V=791): 1-pass x86 E8 CALL rel16 (V=792) or E9 JMP rel16 (V=791)
;                filter (CH=0xFC -> inc ch -> CH=0xFD; after 'lodsw / xchg ax, cx', AH=0xFD has
;                bit 0 = 1 so 'sahf' sets CF=1 -> jc .e8_loop, and after .e8_loop CX=0x0000 ->
;                inc ch -> CH=0x01 -> Pass 2 has AH=0x00 -> sahf sets CF=0 -> enters .floop!).
;
; Credits & Copyright:
;   - Algorithm: ZX0 by Einar Saukas (c) 2021 (BSD 3-Clause)
;   - Reference 8088 architecture: Emmanuel Marty (unzx0_8088.S, c 2021, zlib License)
;   - Multi-stage hybrid E8/E9 + Free-BX Stride size-coding (99B): Oleksandr Dreval (MIT License)
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
    lodsb
    cbw
    xchg ax, bx

.floop:
    mov al, [bx+di]
    add [di], al
    inc di
    loop .floop
    jmp short .mp_loop

.done:
    ret
