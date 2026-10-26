;--------------------------------------------------------------------------------------------------
; Routine Name: dzx0 (unzx0_compat_82b.asm)
; Description: 82-byte Zero-Overhead Backward-Compatible ZX0-Ultra decompressor (+12B vs 70B, <=85B).
;              - Unfiltered streams use standard ZX0 EOF (V=256 -> CX=0xFF00 -> inc ch sets ZF=1),
;                achieving 100% bit-for-bit backward compatibility with standard ZX0 v2 streams and
;                ZERO bytes of stream overhead on incompressible/code/text/sprite files.
;              - Filtered streams use V=257 (CX=0xFEFF -> inc ch sets ZF=0) followed by one 16-bit
;                tail count word, reusing BX = -stride for arbitrary 16-bit Stride-Delta integration.
;
; Credits & Copyright:
;   - Algorithm: ZX0 by Einar Saukas (c) 2021 (BSD 3-Clause)
;   - Reference 8088 architecture: Emmanuel Marty (unzx0_8088.S, c 2021, zlib License)
;   - Zero-overhead dual-EOF & Stride-Delta size-coding (82B): Oleksandr Dreval (MIT License)
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

.eof:
    inc ch                  ; V=256 (standard ZX0) -> CH wraps 0xFF->0x00 (ZF=1); V=257 -> ZF=0
    jz .done
    lodsw
    xchg ax, cx
    sub di, cx
.floop:
    mov al, [bx+di]
    add [di], al
    inc di
    loop .floop

.done:
    ret
