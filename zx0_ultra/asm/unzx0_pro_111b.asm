;--------------------------------------------------------------------------------------------------
; Routine Name: dzx0 (unzx0_pro_111b.asm)
; Description: 109-byte Zero-Overhead Multi-Stage 2D + 16-Bit Word/8-Bit Byte x86-16 decompressor
;              unifying FOUR decompression paradigms in 109 bytes (+39B vs 70B baseline):
;              - Mode 0 (V=256, CX=0xFF00): Standard 0-byte-overhead ZX0 v2 EOF.
;                'inc ch' sets CH=0x00 (ZF=1), 'jz .done' returns immediately with ZERO
;                trailing bytes in the stream (100% backward-compatible with all ZX0 v2 files!).
;              - Mode 1 (V=257, CX=0xFEFF): Enters .mp_loop directly in .floop (inc ax -> 0x0000,
;                SF=0), executing K passes of 16-bit Word-Stride-Delta (AH=0x80) or 8-bit
;                Byte-Stride-Delta (AH=0x00) with any stride 1..128 (packed into 4 bytes/pass:
;                count_u16 + mode_stride_u16 where AL = 256 - stride, AH = 0x80 or 0x00).
;              - Mode 2 (V=1049, CX=0xFBE7): 1-pass x86 E8 CALL rel16 -> Absolute target filter,
;                followed automatically by K optional passes of 16-bit/8-bit 2D Stride-Delta.
;              - Mode 3 (V=537, CX=0xFDE7): 2-pass x86 E8 + E9 CALL/JMP rel16 filter, followed
;                automatically by K optional passes of 16-bit/8-bit 2D Stride-Delta.
;
; Credits & Copyright:
;   - Algorithm: ZX0 by Einar Saukas (c) 2021 (BSD 3-Clause)
;   - Reference 8088 architecture: Emmanuel Marty (unzx0_8088.S, c 2021, zlib License)
;   - 109B Zero-Overhead 16b/8b 2D + E8/E9 Pro size-coding: Oleksandr Dreval (MIT License)
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
    ror ah, 1
    xchg ax, cx

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
    mov dh, ah
    cbw
    xchg ax, bx

.floop:
    mov ax, [bx+di]
    add [di], al
    inc di
    inc dx
    jns .fnext
    adc [di], ah
    inc di
    dec cx

.fnext:
    loop .floop
    jmp short .mp_loop

.done:
    ret
