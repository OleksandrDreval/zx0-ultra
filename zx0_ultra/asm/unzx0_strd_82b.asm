;--------------------------------------------------------------------------------------------------
; Routine Name: dzx0 (unzx0_strd_82b.asm)
; Description: 82-byte ZX0-Ultra decompressor (+12 bytes vs 70-byte baseline, within <= 85B limit)
;              supporting multi-pass arbitrary Order-M (Order-1..5) Stride-Delta integration (e.g.
;              cubic/quintic predictors for 16-bit sine/audio LUTs plus Order-1 palettes/structs).
;
; Credits & Copyright:
;   - Algorithm: ZX0 by Einar Saukas (c) 2021 (BSD 3-Clause)
;   - Reference 8088 architecture: Emmanuel Marty (unzx0_8088.S, c 2021, zlib License)
;   - Multi-pass Order-M Stride-Delta & size-coding (82B): Oleksandr Dreval (MIT License)
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
    lodsw                   ; Read next pass count (0x0000 terminates multi-pass integration)
    xchg ax, cx
    jcxz .done
    sub di, cx
.floop:
    mov al, [bx+di]
    add [di], al
    inc di
    loop .floop
    jmp short .eof

.done:
    ret
