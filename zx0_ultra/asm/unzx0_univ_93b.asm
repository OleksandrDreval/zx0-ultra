;--------------------------------------------------------------------------------------------------
; Routine Name: dzx0 (unzx0_univ_93b.asm)
; Description: 93-byte Zero-Overhead Universal Backward-Compatible ZX0-Ultra decompressor (<= 95B)
;              unifying standard ZX0 v2, 16-bit Arbitrary Stride-Delta, and x86 CALL rel16 (0xE8)
;              restoration via a 3-way algebraic EOF dispatch:
;              - Mode 0 (V=256): Standard ZX0 v2 EOF (CX=0xFF00 -> inc ch sets ZF=1 -> ret),
;                achieving 100% bit-for-bit backward compatibility with standard ZX0 v2 streams
;                and ZERO bytes of header/trailer overhead on unfiltered data.
;              - Mode 1 (V=257): 16-bit Arbitrary Stride-Delta (CX=0xFEFF -> inc ch gives 0xFFFF;
;                after xchg ax, cx, 'inc ax' wraps 0xFFFF -> 0x0000 with ZF=1 -> jz .floop),
;                maintaining 100% bitstream compatibility with unzx0_compat_82b.asm.
;              - Mode 2 (V=281): Position-Independent x86 CALL rel16 (0xE8) filter (CX=0xFEE7 ->
;                inc ch gives 0xFFE7; after xchg ax, cx, 'inc ax' yields AX=0xFFE8 with ZF=0 and
;                AL=0xE8 already loaded without an explicit 'mov al, 0xe8' instruction!).
;
; Credits & Copyright:
;   - Algorithm: ZX0 by Einar Saukas (c) 2021 (BSD 3-Clause)
;   - Reference 8088 architecture: Emmanuel Marty (unzx0_8088.S, c 2021, zlib License)
;   - Tri-EOF algebraic dispatch & Universal Stride+E8 size-coding (93B): Oleksandr Dreval (MIT)
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
    inc ch                  ; V=256 -> CH wraps 0xFF->0x00 (ZF=1); V>256 -> CH=0xFF (ZF=0)
    jz .done
    lodsw                   ; AX = 16-bit filtered tail count
    xchg ax, cx             ; CX = count, AX = 0xFFFF (V=257) or 0xFFE7 (V=281)
    sub di, cx              ; Rewind DI to start of filtered tail
    inc ax                  ; V=257 -> AX=0x0000 (ZF=1); V=281 -> AX=0xFFE8 (ZF=0, AL=0xE8!)
    jz .floop

.e8_loop:
    scasb
    jne .next
    add [di], cx
.next:
    loop .e8_loop
    ret

.floop:
    mov al, [bx+di]
    add [di], al
    inc di
    loop .floop

.done:
    ret
