#!/usr/bin/env python3
"""
e8_matrix_lattice_fast_v2.py

Fast batched PyTorch routines for the 8x8 matrix realization of E8 ⊗ E8.

Conventions
-----------
We use the standard Euclidean E8 lattice

    E8 = D8 ∪ (h + D8),   h=(1/2,...,1/2),
    D8 = {z in Z^8 : sum(z) even}.

The matrix lattice is E8 ⊗ E8 ⊂ R^{8x8}.  The scaled-orthogonal shell m is

    S_m = {M in E8⊗E8 : M^T M = m I_8},

so ||M||_F^2 = 8m.

Implemented exactly (up to floating-point tie-breaking):
  1. Euclidean CVP onto E8⊗E8.
  2. Euclidean projection onto S_1 = W(E8).
  3. Euclidean projection onto S_2.
  4. Euclidean projection onto S_3 (2240 multiplier-3 image classes).
  5. Euclidean projection onto S_4 via the complete product description S_4=S_2 S_2.

The shell code is pure PyTorch at runtime.  The small finite tables describing
W(E8)/W(D8) (135 elements) and the 270 multiplier-2 image representatives are
embedded below in compressed int8 form, so there is no NumPy/SymPy dependency
or expensive group enumeration at startup.

For shell projection, S_1 and S_2 are flattened into W(D8)-cosets:

    S_1 = ⋃_{k=1}^{135} C_k W(D8),
    S_2 = ⋃_{a=1}^{270} ⋃_{k=1}^{135} A_a C_k W(D8).

Thus S_2 is searched as 36,450 left transforms followed by one compact
parity-constrained signed-assignment DP over W(D8).  The DP uses only active
subset states at each of its 8 stages (maximum 140 states), rather than a full
512-state scatter table.

The full E8⊗E8 CVP now uses exact branch-and-bound around the same quotient decomposition.
A cheap unconstrained E8 decode gives lower bounds for all 256 outer cosets; a
one-column syndrome repair gives a feasible upper bound; only outer cosets whose
lower bound can still win are decoded exactly.  Their 256 residue branch metrics
are computed directly from the D8+ parity/mod-4 description rather than by 256
separate E8 CVPs.  The remaining trellis is sparse and exact.

The older dense formulation was:

    (E8⊗E8)/(E8⊗D8) ≅ E8/2E8,

with exact branch metrics from E8 CVP.

Typical use
-----------
    import torch
    from e8_matrix_lattice_fast import E8MatrixLattice

    P = E8MatrixLattice(device="cuda", dtype=torch.float32)
    X = torch.randn(128, 8, 8, device="cuda")

    Y  = P.cvp(X)                  # nearest point in E8⊗E8
    M1 = P.project_m1(X)           # nearest M with M^T M = I
    M2 = P.project_m2(X)           # nearest M with M^T M = 2 I

For numerically robust exact decisions near Voronoi/tie boundaries, use
float64.  float32 executes the same exact combinatorial search and is usually
much faster on GPUs.
"""

from __future__ import annotations

import base64
import zlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch


# -----------------------------------------------------------------------------
# Embedded finite tables
# -----------------------------------------------------------------------------

# 4 * C_k, shape (135,8,8), int8.
_C_DATA = 'c-pO*QG$ao2t>Eq``=hyO^QYe%z*VrcPmL4-jG_~f1l%ByO>|?*k1E?9Q)P$Ik$`Bd|b5ie$mg{ya%*nmbi1AV!Xs%<8blYYiajyOOMOOdwCDzq~_1YBi+YwqWLAx(4n+`c7B$d+c;yqtlO_6wSJv{RzEL)xc;PXWE_c~)^$3JxK5w*I<@mO^>e!p?(y`i@*{Cp9V+@G{aF38`iX9%j*NcU{E_}_{+|BXcyE4IKhe$N(9#d-FQ47UG4wz44*CG2e^x)Xex80_{UpxJqfmcTzq4*ZA7SdJ>yPTk)?Yrm^|k#t@>qI4OV6uTe~+JR{j>FB`FqNb7vJ@5<yz4`;(gYm@jevs<MEf}M{{GYqrh*KKb9ZOUzQ)luUCJT-ztCSdQ~10FWg-m=J|{GMf@OsS^hM?S^g0J9)B$VS$_HbhvOgd_r+b*#|!?c^3U>nulp`L$uHuc;n(7y<&X7`$B*Zq%>0c15dUA?M}4H=->Uy4&U`L2{FmjQ_3u;vvGIETtbgux@8C23@%)GU<N1%}m-TPtpV5C=ep&ys{;kZ<`2VSY*?2ua>%WV8;yEJlU)Dd!zsNtxzg7R;edphU{LA`J`#0-9<bRJJ?cXf_Z2v(1zvd<C5%@prAJjjPf4+XFVV(nn{!v+f)PK<WQU5{lL*L+khcNmF>OXA%$oe0@=dkl&_ixsJ$p26M%f?suXZ%~;|9|Db>i%E#&lwlJmq_>r@$dNu@$dPk%D>0Gmp|h7DSvFd=8vr(;*aH5^Y<%$*!pXJvh{cCPUvUV@2UPz^<(qb`VISAxDV>zpTBj(--E3D&pyYz7d7(FuHSmEz<duF_1kale=q(2nfQPFY7rL'

# 2 * A_a, shape (270,8,8), int8, where A_a^T A_a = 2 I and the A_a E8
# are the 270 distinct multiplier-2 image sublattices.
_A2_DATA = 'c-pm_i<a9U3`7n8|IN5ygVzxZGg8u??$$|+c}SW&!EQc|qtu@xuMcuRIKV+p={S%2H=V~>j{2g#F7?-$+xz*fA072?`#g`&bA6ri^?tdpRK``k{g1OpJM)Iz-W)t<O>dZkI-gX3moZQ8{QJ5+>*#DhwR_^=Y;WTCwCmSl@f*uj%(}7lNAZ`}U(KJ<&b)#0NA2I{PX^;a?caIR^2nV(rd=^{Q`_78mE*7G|I@Bt$K?Oh?~~8p)sH-Wwf?~PQT?g?+xp939pU&#=J#}q)}O84X#IKfU#@?#e)H#a^!i2hKTQbN?;WpEk6Qks`d9Nu){jm7r6WK27txRRyrcSK^((4B?)p*nkF8(7j$6O1ZUptu>St8{-0P?6uc)6N^DPejMEd)*v*Xz7r&s^%X(iZx>YwP3tiM1%x&G$$1dV<!hbidqsGoNnJ^DHOp1A(Lbu6O4Tm1+5Z^!MeKbaq5|5Lxc-L#H`A0GW)@x!bCcHSk}e(Im;hsqzs56Yk3&d$S*f2t#$A1i*j;|DpS@x#vB&Lhk}7C)7L<ogTe-#&h@Uy*;7UvB;|{)zmtr~RIPld~83NBH&n{`IZ{Z~a()dGcT4AIiTv#WS6$%`~0$q=wXb(B^j5KY6^L@AIxLtFv}i4^m2ypHcp0z~L}|N&U+F{+7Q^eya74`TN;U<*%E6jNbzPR_k7V#eb2%Hg3P?zm22H{}{i$`eE0>!_O`M)%ctIlly1>LFKpAp`ZH)5P#|)Nc=maEZw5~p7_uBE%SfP-}bBG<ZQ1hUQRE65Pzfm^zw)DTgX3&|4x2C>(<v%<+pqOjQ=8kGx;9;wSN%!yY~<4`sMz?$8T@_GycOm*tXC7KBUcGv+ox54-@~z`r-Zq^LLduQUBo{zwBT3{=vq<!_TdMQ1#2^anwK9@vHHt{zHsE*hiQBhj;w4e*yXf?mrdB{pS^bReo6hdig>5?aV*)JvhvNC%?V?(ftGVk9+?>{G;&?8SDc<|G@mm)2?3!_>cK}U(A2F<0t!<J^!hH5&cJoJm&n~`VYGflK;r~K7!~UJpN;}lb&N4?ZN#!mEZ5{?(Snl{|WTV-9Nbx&iHNT5$1n@Uz&da^)o#;<MR(w{QjH&SdSm=yUXJ@{PpTD<u}paRlHXF*ebuh`m6I_%O5ZQaosrm1M?pu|MB{1{P*%h_78v`ga0Gn%9s9O%75k`WdHHk?!M05-s-p4f8c%NU;CG+|M2;T40-J1KYb2C<UjKdcihUNZpHnJ-mm*PT2nv8^dIpL*FLh7->ZCy@E_`+m;Yv+rg4P*2jFk#-@D#d`^YLkz51i`cdtJs%HPF$MfeT*@8tK2+iMPb{ll01N6!zoP_(eL5Vml&<M!Tl-2OfD^*UWY*Z<cz8m7|qbIAUI@gMpJAHO005x+O>CH*(~UjqL1I-fj;!2ARBA5Q;(>wjIpR{u-z`31c1uk-hPzRF+5PsZQR`YmUKpV#+os{<mxC_kBgF@Cc9S3UpB7yKD{x4RF@_yzgr<kyN@AL9t~6Yy{B`vv($`6<Nz(($4DodaI|hx`@!$@vTXkIdgFKgsilDnFlf>+2}<Q{>-*pNf9U{6qYFaqmM1#QLTDB=wuxXd8XXIih~;Vz&4DF6_OE`TqPqzeOfptAO<1!;iP_j(n2!`^D{u{!#rE_P<(w+B}x)pXs-#zmtAD^_T1aT7Px@$NJsYp#=5Q>UMGNzv=qF-@oPh9oOHr4srce^o#3nUvC?SzW!&sO1paN^}3o*KStb3uzqaaK04s2|E+$I`h)r*)}O9Fp#GTtJnLpT>eZip{d4_^>&M;BuH)bS$6Gg^JW}(A<sVr;6t}-T<j+57@w0V!_y5e>&O@C4Ki8YM`2'


# 4 * A_a, shape (2240,8,8), int8, where A_a^T A_a = 3 I and the A_a E8
# are the 2240 distinct multiplier-3 image sublattices.
_A3_DATA = 'c-pO<*>)Q_5=2{v*8hL7tfS_lB3TIp;zq(V4=vfwT2NSXiTmfjzt{g?|M%zd!=LN_|9ku4&wsx^zWwm`{>$Ts&&$ux_s7T2&&T)o&)3h#<^Idp$MyHmpX*;;zP~>%KbPC@A3v9$&)Yx#y8ixgxm>P)bh%&b&ws~%;`_gz`|%#B{lxC{2h@JFdt5KIpV$q5K<umCt8Fv;!LIzl^`l-sK0kiGuHW_A72@xI^qZp}rS=oM`}I=$(e81*)P7=jzg}uz?Vi^Q?W^5!y~F+(cgFyX$6v8~UN5w-<Dd3_H-EVPfBF3Q{<(ejYvWy8@Ya*AP4xBgeQUJuYpY!w@$>s~`~CKppWoNN|Gxfy{Q{r2?mmv&=Xh$(U+Mgz=0DtiX8v^Thv#20e;ihC{U)4$hjwqx-;?+LeXrj7k<Q<0{h{->UB5*8*M7HHhr;#m@#EWbwbsvk{TlAuT0e*PJ~eP-{r2r&*Wc59yX$v-@7HypeYGnOa;*EuJ|JDctKD(E)V}UN==$sSFK+!k){WOXlC2+g-F{ooTl=SU|DyI^wEf)v&9xui|B3y}qg_9L-ug|n|C_XXYyUWT@89p*+rOs!N45W@`$xNf7VTgA-C`dc?!O;)Z_m}*zvuhs;l8c?`|#e!b-L#V-~M&~f4Xn?{7~QfxDK{2cJ=wMBmes2^|9^`mG8fYeSi4_w|`ORAG&{Y=O?#+IW|4V`kS7g>U($3U-h|P&DcBtrRO(wezfP0;ry5F|G@ckx&K?7Khge;o?q4et?oOH{RkaD_5Ay?JFb`6Pwej3OYN)O^Ln9uwL7l&nm>NW-``g&|Lu%Fd5)*&_j+!AK3QZxv4iLTx&MNn|Be4coPWl1pY&t2{3l)i1s?y@|6<;Ma*k*72lU^l`NP_q^WWdMnfrYm<LmF>|B3uB%74Vy@3sGluHV*wCBFaX<$h}Ux$Tec)78gySM+uN?{&`mzVBl{f9^Y@>z?bnr~1B4c(<Q#6WZmn7`M=WHTR!!|JB<6=KVL&e+T_n3qM@s|2zHjGNC5^!-oHE{{77RPd}6Vi-!La-rwC<`Jbe{nSYLdJ&u38ew@~w4gcr#y9NHwssA<gf2w)+c|B-f?ehE2-`o2iMZ01DtJwdC{#W_kOG8@wYS;At?)q=Xeq`*+bF1-poZF9ce0qMbe%HevOZ{)k|4jX7&VTLvM~DCM@PDNK>-yeZ|95@vSG)H7@2US>`R~2|y!ziW{~PfCFa2)||3C7d(fEP#pVf2gah|2)r^X-B^Sg>)(DS>Cf4K8Ij-QD0`!W8G?~uk%s^7aCf2rU7<yX(_Cw3_QqT)w1{$k@#!v9?T?u$557{6lve;5DC<4@WA1L9}P`3J|}O#4sbIB5Q&@jEqtS$pIEKjzK4j?(criT_3MHx)mm<Ilw(>HfjSFU9vi{04^~A@v_D<_*+-Dt=1a&&6L|`{DR4j9)I|uu=T?&F*-fR{XdTzny<S^Zt{0XZju5@oRX0cV88Ml=f!+x&HEX8x7<5$LrT=-P?$NPrq9b|DML*yYX+ej;8VZ*ZqI51MRCFeSY`cy8oQ}eLwy$z8jAJi~UC!|5yF)xE{7IcHR7g-TZ;tkF9-qZkK=XJa_#1GkE^iJb!_npIrWfJ3qagZoht<F#lnQyEpSE2EWta!Hn6Db}0Ws<$w72P0ioP@&`cv$kKng$R9!RKbn8y;(sa*N8@kR|4Z{nRQ^jEzf<`$Hvj)R{_YF^*L~pK$JcGHc<hhA?~})EPPzS?+gx%RpSQijV}JdMztw(6-(ThLr13YE|3mX<RQ^wD|K~Xy&7Y%@KZNpcCiz2A{*B8&lH;e$8#Iq6$Y0vZKaz30H2+D&KPLU|9B=C9KcV=?V?F<TTR*e>r|Ng@=TB9?<>A-O>?d|Ge`=op#q+03{+RH;j(%sE2iD3z>*jwA_dVL9o4;n>f0BQ8x^Fjst+v~79c*9h()>4?zoGK5X#Uk$_twjQo5b^5`D=Opm&w1g`CmgGOql<*&foL-U&0>dU)27S=3n4EJllWfd2}>?F3o?S`G<}CJGOqO?N--sXK(TsChwoE|1ke?H-Ay(KS+C;|8Ti`Yp!R4$Dttgzt_B_xARPN{;qaBe{!C`nV!GZIgp;eZT_gKAD*0By7{9~{-4WVZRC$e`G3_u&Huysv(5T#H~(3Cv;Iu-XYb?g7>92Bx%}J1?s*-Nf2;Ed<NVt&|8$-I+sMBS^H1fz2j-u`{9}0kJpa_#oBjuM7h3tlLp$>P+vR;h{_UheTJJw?N18vJ=1;o(VVD0n*u(Yz&Hs9G#N+*|%wJ~t!`JiQW6t{dx_sXQ9d}>mBOmG>5W9ItE|2s1&9!;`^Sh7p<2ILle7bq7pT8XEAB%Yl=N~Wgr)mDI+P9?n)9Smwu9M|oZ|6_r{AKaJVg9mQH(>tsWZvoJPpfep=0DT?>ES!W{OQAQ^Na^V|D(#krv5{hzwP{QNB+_ghfVXhYu<1-|GUPGUwO%yeZ9ZSp8)yi%lQ8y{~YD-(foCnzo+u(X#QNyzo7Z&Dt|uBzf<}5HvjxL?svoi)A*n2H}w5g{(qW3r|Ji2{=KRnNbQe*r=Fuxe}MA$C-nzW{=TbU(E0neeqpQrK;_ZJ^$#NdXVP!Z`C9$@2bBMF#P5#vC$E1v$E~{c6LsBt#Z@x<i5;q+Q1u_Qe!|v|i1<~FyDjQK!upek-4G88>sRvnk>S3r`V|^K#Pu_#{U>o`6hEZ(H!6N;?M?jf82@!2nC>6y`bq0&RQ*r7e^B*9G=JaLADR69!|(W(#~<bI>pGlX{ZqRCP<1#of8W+$!TO^`9aU7nb=bW<Ppkebt-q4r&%A$K_tkf3)SsdHC0oA+pMPj?Sij`z-`x2}+Qa&RasJ8b2XGykcz@MzX#Je3-|N=T(fR>ecd6<J{JgWSAE5Pvy#6iQ|GWA<xBpN3P2K-ndz1e^X}{?FK<gKG>kn1^fV7A814rEd$OA~@|F!Pu&7TqZzp5RrUsLs$seXd;15y8(t^YLouM@vTxBfG#-*fe+jrz~1ey`f6^?SJfwW+`B=1*&H*566}YZU)i^|Nnwr#t{^PxA+G{qC>balOg@U(_MS^}A93clE=K`rWAiFZP|N|Ks{)c>lEjTYJ<0fo^xJ{&;9dUcbA%FR0%=+vE2i_Fb*|W!Zni`sK8K($z1!`p3Z@?_Zwrqx(4~tzXvl)9n0l<SiU|3~By?oA=WC=|vrItA0DEe-`s5u76(EfAjj=`M$4Tzg@mVvwr;b{J-(8uzp<cKdyc2y0WVJe?EVheq-vtZT&f}|6ccdU;jP-UTOWeTDNKa`DEVj)t`%bFs&b__2;#p`ug)n{OcKq3G2^?ypd-8`r!9}=B0%8qaCV$SM~QU{xjCkXYn6UzrT$CEb8}B{D#*5tN6{4c2w)Q()#Toj}F)G&-Dk=`fa6OK<l@kc^kj^9BKXbkbe)~U+E{L_1j8+f!0sRe6wj`0M~Cw??1V3OTQu2Pngu<NBRj)KSI|pSpA4C{Rdb7h4m*y{^z9MMfF=$ze3h;e?At(+v0g!JJbBDs{4@EPpP^q%s&zO87Y5ZQ3pi%F{k|JuKq@<|Dfu|sQv@c|1jf6>VLrT&-6nif2HV$9R2p_$2NaZ=#RYF-LIG0SG(u+lzvH>_q+NfY5!64KJ7n1KgH;WOZ^ncUu60#UO!~67c#t$)Gs-|W><e@(*9feD^!2P>c1HMk-F~4dd{YPj9&l$I~5Rj^<NhHF`@p8sJjX6Y5fS*e{ua^zPG90c-1eIb2R5~F7;!4{^43bhU%A$y4^kfn5x6l)qlzLQwIH*Q2%7DU*q*pguTgMyxf<}R~<~MUo+;-_VjO*eom@?Gxodk98LY5hW^a?x%>J%r~5zKo7xvUr{7cCZ9~6jI&Ql9J+(cpeR=PO{!g}lHu^!X{#xmX;Qg1;aapWCMZYH29}?X3OutC!AEo+5<GHasM^itkp&u0L*T8m@`ZdBH>(^Aji{TiC`c1p~OJ2XG*u(nu>Ntt@myG_C_1_M0-$y(+tzXAFN>D#4)qisOQ&hibocDVAQ?!0O(0`g2h8z7PSHDg5qq_Q8?)*UYpQwIRT>oA2M&ER#Q2yvde=5p<bNW>-e@*H3y#Ipu+rkfv^_$CmH}tPU{in75Rzv?P)PFkL=lV}fzstoxq`lJ*EZ?oG-(~f`a{Vl%--GqTxPDfo-(&49zxUj4sE${xKNjkT(fu3M4^#8Ew1@jAr$6TAPpUt5*d5nV{NG%E3hIw>{jE@c%Gg`|DX2dd>W6XtDW)H$^<#uR?0;B)EYuI1y#MrIMfJl1{jJ(=VjY0`V_p5QP=CnjhvD}h+k5%_e?IQnPPe?N$21b=&kpn6^Bl;|e_H>H>4!P}Gp8S>_0JyfeYAsmf2y09>X$Y2zn1!EyZU8a{jt%_73z=0`d{LCQ~fWsZkYSa_if%^`!O^9Fg1T1@vFlRoa%?wyvwfsSoK@K@;Wp7dVi}w=Jn5(@vDXY8H!&~{WRtOKk|o_4hF4XtaSiX|IF#PQT?nV@8p?}66$Bw`bYZwPCt&;pDO*h)V|tL{WhmRr|K7+ew?cx$NF`meqhq>QT;d7zmxSFzj=eNJi;u0u<FL8_1CIyo9e$Q{k@d`yr?6l{Q6V<P*=Y%)sItk=Ttuq=m#$AZx;H2X#7(BL8%{C^as=YL8V_v^9P;&q01js^JZ87Fzr7;KXLi}7y60l`&0czryr>5&4%~M^%uAF6RG~7)o(QVgLU4QdJIkd$5g*jtz%yQFsr|#`iuPev--DV-&)TRY5rW@w^RK@rT@t5$JY9hzJ3hqPYV6MSDbCGL*e3YDsPSJPtNr(x&Gu@Ka=J^TK!F^Ke^D+jPyI7=d9QBwDdny{Z0A(%=_1KtNIQN{ZXV}Y4uCt^AGI}^(&qJsnf5N_Ex`gt$&*8kE(gI=H)qin!ic)Q@Q@9&}|L%Pig)q)?XF*pVr>we@@yj%KxPLt-JcGD*sd3!~D;ipZw<hIr2an>(@#5Fg-t4{n%8$RO!#s^Rv^hrTXVq{}$?(Ep%)n{amVF>hyP0{o7Q(6zKP&^Sjmm4ey_x|DArY(?65;Q2(sv4ZQLSnEqMSA5QhplzuVQKePJBQ2%V9M~d`|)A8f<ms9=Ybo?mYDXRYq^q-Oc#Og=G=TFBU)t`3yYtlZ{|83|`Pj#D8{pzY)o9b69{p?h~x_<Yres+ED=XJXJ-B5qJ_8T=0Q~htMKRwu+{s;8GBmHhUe>C;GseUo-_e#H5+C%-~>Nj)yxKzJ<SAX2;7Z3JO|Ig`{gYm!AFQ@Z|8n=D@@=QNmuiH#N-07#M`r%Z6k8&4Md#b<Z*X^GEI@RxU`xfWF$NKI2`sq{sJEuSH^zW$tdP6@Q#t)c&I@MoK_48`o%3D1R)L$R;+oSp!rT<RPPfkDHou9D&yf{B8-R<LCO7-Jw9zv=gukt8p{*Beo7yFkQXI$vuNBa9zKilc|H}vx({p@O=>Stqq0K9*ypKa~^{`vIXyZivlAISClk$yMk7jXUlNWa_KoA$5$j&cNI{z1qupyvn5FHq~Bw1@lmT6eVI7*KveT{lm4Ow^v%KVg1@^#7ghC(qsDHw^1HynkB1W$lYyz5j83o5UX?{)6R5<opFRe_{Rv=PyL_m$kS4cj-52@h8&x%knGC{6+Z_YW|Y;aQ<4HyITAT+5bX*#iadC&yn&g0)9hn_vtuR`t^POh2dww_FIhGjDHd8zdQZ~ZvXLnBmbG(H!l1yh}%Q?5$fF0=TBgM1LBS?_zg?`#V)_1&yP6I1F3zrYw#y_`4Nc!AnjxRgXBj*{=>YzWB!9!x842K_v+qX#W7QU#KM0z)xG5WhBbeJ=6?=#d+`2_UqSf~GGFU;y%zr>z5nFCE&fHypP2A7BL0NqZ>an=z~9*7XSn=9%<mBSgOh$2<^NIs2j+Lg{E0RHgX*_S{SV8JSo1^D`{(=!%O64d@e}??q<`u7B~CwnsJr^D%cA=6ihq*ozb`l@j9(J=AID#z`tieUVE+O97Nj4qI2u&H)bU@O{+84$X!2i%_y2pAy|et7Vwdu3)N|waUkI8f{20n#vHTjtUm0{#6vwH_ze)KuR6ic`ZybMS=r29~OsXGg`87sAvhr)v`d!Drar3w0mQejg$InUiQx|%0E&dL@|3b&E#owXx|AOCx=6}lnS?HfmbuSzIpD8~m<^QDo5~W+&<p-twlDOT5`HS&~EI+91`?~y~I{(=E{bGOoJ2ekA_(c!9<2syQbh4w*@1Fa(&U5~J>W`hzPci>U@RL&hQ96FqzNgDiO2?1mC%OF-<~ND)r}jOR|3mpt0l(>F=jXpOKd8@dlKiKL-&EUuc#bYVYLnkY+uz-{!Eb`?H-B%+U$Xos(SDA9MD-6Q{3GZ;Wc(yu|0?-OuzsBJmmGgZ@t0D5(%9~Wr_|&(HTXv%e}{3WG=Hc1Z3Ty`$&cFQKS_R$;XhgZLBoH_`CoGmQEK1iH;v=2$B$CF2b5o=bTcUbX`=hE<Uft$wa0%_@q;cuOV#nG{3q4_Cvk&_e?$3EyZomvzX|vcmi(riA2a6&rS@I^(&%^S@t=l%5b}$1e$1R-#Q8C6ep8oUW%)6NpJe-w;3r}J5acg8|B>VH@bS~*A8Bq`kH6IAA1VG)U?1sc2mB<&UrP0>0sm;JUrqT(O21mwEgf}DlwYLyNiP4q)_v7_vR(cXy}#oxsr+%ped+R->UVzgZ$|dj4)B|59011O89&PV&*vPYp#9eTshr;g`9qLj1@E8phm5`D51svEltY#Br=))V=VO3<|9sukFK>K^TQu(OH~#wmxo(ngBGC{3{B^Y6e8zXG#?9<{>cjjh!#`TYts?$Z%73x^D#L%F{HcJy6!Krr{ie&Wviz%@ABOldjz8trZ_2O2?H}@IDF14kUlsDFIKPPTrviRc*nbVbs>`2h@{44=Cgnek_ix#2j-ur!86J;$e=!d1^Xu;}#&2COp7E&P^CaW%Qf^|-AHw_~vv0`wL6dXDF2AQ`@6JDtzeDrq6@Q2FZxnwgwXb%Rf8+Q)%Kz#3J1%~K`9C6lF=_WG|C{oIy8Iuizp3~?seQ_?q5L4nzfk-j%AZjDAgaHpI5t#&(D8?yeub=asvM*yzbLr>>AoBMBBXy-xd)1eLHR+c{*mPm8U3TmA4>B-9lyx+f5n}l`aO<+l<K!F^f+4lBzpgaj!282MD<S>{3SGhP=3=wziX;n)ZjNw`A;doDdpEF-Jvf3Ddpd!?N;kYDL=}sf5&)VaGVgoDb4=`{HSIA=7JxE@;51eO66}l{#443O8c+lM>+jd%&!vtPp!L@pG5gr0l(^G*X38K_h|F4Hu+QZ{oQ>Xe<|lDt@%wj|F4`IyZok<zvK8(DZeS@?*RUksb7}-s3?EZ@vC(GwXrw+o64_B`2mi9<>H?zZ;$hv*8HrLUzPKl=6REx-(>k+hQA~ETT%Y3;}E(0U1=ZkmlhnXCI3qC!%}`%^*43-VZ-|#*Xi=hAb)Gfa}@nJ<)2ah)?g36zy0ozUna+2lV3*pRg_z&_*K##@~f)f(Q)`ve%mg8&GD-Sd(5wL^FHIJIsO~vcd30>%70V+=h;q%Uq<t{DSs~I$GJS{m_N7Af1BoyJARtWAFp}ex}F2Kf0RFN`Ew?Jobux+f3Eh^ls}j5KOMgg>_3<MI?8{mez-2bF6B=-eqPG2OZijPKIKm>_<6{GO8I^6{4VXGes%RfrTjg`{|o%DYknZ)-=+K|%pVNzPx(ufUr70TDSyfF1D*d5^AC;x8GQGpetGpDQvMO;C+_kOoqqaY-_%cEa1asyu<ED3tw)i4wWIt*xqce`@|y3i*3kw(@?lr>%mM%LWY^_KQvG#lAL_5C{K?8m+Td4C^>+h)<&@(U@GIB+&5&P7`GJn#>DDjI|Afz<u3wZNO8J%P`lWQcyZq3UKZyCIAwM*YpMie6)X#Suv|awF)8AJdt(5<l@<Sc}H01|2_@zsJYL`DKd5$4}mGVm`>p;XGb^KP!pHzC+l)ozH?<Rj$&BG}__ON^ADN_4lNBN=d9G&uKcloiDA6n-fHJ^g-KRrh@__GoJb+K+E{;S)!Q2uPnZ*=_G20u39H#&YS?my#rY4K+%f0goUUH+WoXzucF-S=~TzdpZ~@n0Q3*YRKVzPHEE{a5_gVO@p%SIQ5e-1=R9?=F9rwqMFG3HY^$f1C0*0DpGL-=O?iH~)?6<-g(AHu$fR{s89RN`CCwKIQLG{;l%gSbO8Yas1nqKP&U^J%6_5!K3&G<>!L>i@Ck=A7FlN&Ywm6H_YGV{Mm>fXzgM99e+<bxG}#s<nL1d2j%ao`BU0M{{`mvn)uVAZY1LGru;$6?=}2E%HIw6vn~GMB>x%mFVFqH%m20f;GDmV_=S$&>-H~{|BJt0$S=hF;GDm^)Pack!<@ex=}%aDvwt}3CvmeDe^~CHLjLgNyPcjR<qy|=q_r>i-R2j2|NoSuxxw#k@r%>>-|~;m{7?DC0l&93|Eupu<42aCob#vA{O$O~ZvLkHW8D7X{Jr4+hWup1U(Wfz(|&~c$z%Js>=n-z@PC&)Tg*=m`NxZK&-llB+_vmp|E2unh5riik10P`^_M<>81sM4xp9;KyU*`k@^96=ne%rgzjv3v+p-7#zbQ9?^JCZiU0T05;qONEgO1<p>bF$=rh9+K|Bd;<;rmm5u;bUN`mZiOc&Ohvu0#35#V+L^tLMh=FTO*{AEx|X%Re^!-l6WH^n(Wf7}Y;v{_!S%nBPCE|ET(Hqn>}rACCF2YyPq4XG#7s<gYS*vev&9{9?>sHT-4EUlsgi%wJ{v=J@?Fe|61Y_WV`Ie>VKSLH`}|R~bK=^8YHwIp#lCy_J`~Ok`i}9Di@ZpN{x_j$cjreO-RF<h4P5wc=;D`PFd#u>5T^e^}m}nLjAM+wot9$87Su)A`5oyD7gA_dCHaR2<|Det5)xw!DZY|9kizUH<rCSA?o9udmA=7w;Xkp9K2-?eNQ+{BX)IA9yKU{<wK>*M9E3CBNJ7!`1yq-NrMAJLPw$`fCgRH(Ea_KU}V#1Ghu+C=fq9<%c+aIp=>59Egy=z2=`&{#b*5KIOKg{PYEfBI2i~{9(XfZ}HRV`l+}vl>bZl?Th*2#1A>RAqPjq@k1PUh4O<dM=#~4FZk=r?{{-v-+Zah`7h*`SKZ4t|6K6XQ~r6KH;3oQ`RQBy^SAep>{YuR*IB&(>Ag4j>(lp6`RkaUE)D{epRV@J|A4<9@zYcOHs-fGetI$N^3zj(x8<*g{O&&geE4S}|J?A`<^CP=$2a-kPCwG|zjyiFDStcVXFGm3<wuWkryjq1;9!UMR~+k<|84off`5(q(~zGn`5%IxP47?n+qC~F4)iX+8{A)T(_{YjRR1mDe~Wpe=qu*@?Ipk4@wa1sw$Od+@~>0=H0D1;el_R!#{6l-zs~u+h6f$;vv>K|G5?wJcLo1B;5Q?FH0^&eKN<3;Y5!~Rn-PC{mmf{-b-!o)X6-kC_94Fv@RyhTGFA82<uBKK@3;EK$iCVEeskqNBmF4Kk5=^ql>e;khqz&%-#q0V2K?r!A2Q`PtN6|P<Us7J-7Y_RoBwS1$%_BH!9U*OKXd-Bu#fq>hQlnLAM=-g{icT>_3)oA_`eZ9+47fj{x9OEI{vZar&4}$ga3>8sg|D{@{d#gs^b@X{;lB;clp7Xe{1&1A%Avazr2Ct=f?bA{od0DT<}w){Ab5sb^a^G0j2rJj^FC|r^<iy%1_<nzXta|-FJikiu`|+zuM);I{(*#zl!u1D8JR|-zxnT$N!}K*WmsU|C91#8~oRZALsb7jvuFZgRcE4f7a<gDh}!fe>UQ;$~qm$pRIaLU4E>3fBO8Y-BtY$<<H9TQ}TW)$2H}@rujn)ek>Y4ls~J+PpuPN=#ixS*@&O&__YiD|C4@PgI{~9`|I*=7dj>p|2E}U0)B3be@pdumF^McS5p4&V*EaY3x3a`Y5tDW4f6cZx$aNOzg_Tim*3CvXH$M`nm^<Cu`d4w^J_)^%jEf^{1?i<P5HH5{%p!$RqJk-ADi-1Eq}JjPv!j4HGkFD?@InEJiju2tIgji=f0G`s`#&6{wn1kVty>-x2F6<%AckD)s%ne__4lz7V>Ll{ps-h5kHXfZ+H2%ia#dpAwN*@Zz;dU@pE1LYQfJ%@hi&TRq-dqujTyHHNV&MPa*#o@^^FoDdPuI{%y`bwfx{M{;%hkLjJJiFV6GMIDeG!i+B0Mj$bJAtsuWw<u6kHuHzR|{;K3vLjJHE|4sg|8pkO=`Dyoi9c*9hC_mS6z*GM6E<c&_bBA#*#xwZ-({bG3FGu|2#k}9*A8+%UAwO2}i&K8A;wT6GZ~5H<esjc+b^K;J|0<4d%5U!Si`BU;<u|KyQ<vY|<xji(-GG0*?Ej8`9P^7|{x0PY>;CuP2Os<8$`MZa!|D0K@sB;fSn`k2`GNA27v~4WPj>uWrJGOr$Ckew@{1{d(D9ocKMV7p;q#~bLCTM&{N$8BII)lTvy>mb%YR<zU(W5Je!b&IQ~t8$PaFMn<$tF9Wyh~}`3KTI<S(cEILg0P{5WfG_;D`pq|3ji{5Z<Lru^lUALsbf+x%>!f9m+rRKLH*PrCf<l>cV=+a`WO`PumU$NaauYV44Q-{5agxkUkgd&>U~_}goKc*x&Y`@obxe%L*))8&`L=db%l)eb5DobtN|`;gy2IpP67y~RIQ{r<R4%1__rpF94;VBh3Fr2KNsU)TKEDSy4oPhW7fTm1HbU%uwQLw-ihUkUi-OO7VsU(fBE{PHe8KII={e!bj(pZOVG{<`gVhQDt4?~uQq@_Vg)lfO>)e~zC{`5&&|tNoYbuLJ+V)IK=>BYr#O=QjA~Q~Q8_j`;5>e^>IDLw-BumpgvE<Cl;7^fk{q<<B?x@zZl`%8yU^?drVP<+nHZ^UL>6+fU7d4gNXemoNMl5kEcU|0;g?Ha~rzfA07ph##KLKZ>L6$M1Um&H3Fuet62ScKmSKe(t@y{P11=ch^3B{*<2s_}7S^?fBQ4|H}B+0slGDU$FL;AI<pF>ikUk)0E#l*mHjKl0Uu6k9PcD#gC@XZ~4(7|C#a^DgQa(AGi36llW`I-?bcP!=JYN>YV?L_<4>W?e-6pKaIa%li#?gLpA(S$xn{>)s}yq^OF(((ebAp|B>>m8~kL%e>}dM;`q7o^K;M1xjatb_x^J^5>swk#<&08qbirX7}M?f@3}-Dw-lD2grA-B&s+TKlwaxi)iFOi<X1ZW^#=dB#lM#P?2vywi94O1BjsNQ{O7vwsd25-{cqpz>)yeK{G*7UZTZ_dzkJHE#{BHD-%RZt|C;*07yjlhzd7PxTYh%VZ$|5f<5#=&gYvH%{ARR%V1Bk~KgrJy`Pb?E?fBK$e?Iz~(EiEtt3&?u<o&1Pfbyebezm!;=2y@CG+lnQil@2uryO3)-?jWyqkC@osf^$2>K7fqx6R)*?=Shg@ck*jH+cStze@SPZGP`2e|O;FcKN?YKKt)=EWfsVhnW8>-aBZ&>34MZciiYMzn4D0yKl9B@kfe&Wyrs+`kyKPR`GXZ{%y>E6#QMxe`Ne#&HoteL;fM-|F-$PoBUnRKZN{W$v+hQTFgIW{9wtyD*UG|zqaaLrTkjO&yD%Df`2RZTOj{7<=4{r!|`uZev;&`8GfwP&&~O@5kJ!MPeXpB;z`B)%gKwa_?M2q>Biqn2QlMUj=FU{{-xq)ru@rPKWbth={HgSrW?PNUls5(i|%@lzq!yckNBCXe!JrhI{kLV8<hK*CjWE6&o~*z{7u9EbnQRt&YgA8o&KEThtlt_xT`7uGv((2e&rT_vcZ2mjr)}UsOBHHe~|pdqFYD#iTL~det-pk(eR(={5{Ig!~8zMzteU1Le~iL<97LN0}l-Hzees{#vhX$EzI9j+`PGdfaVWD{uAV9<@_PcpF;esUH%nafBrSU3Gjyyzlrn5di*5CZ*ul3Z>zx{n(~iQ{*bD7dR`COSG!&QQOY0M<p(wRKZ1Xq@qd*6McseFulYCpn2`UH@;e;Ah4N#D=ilV7Q2qztpS1WT6F=V6Ug@?{{srZiQ2t1If66b}<$q9n#T}vijg)^t`5h_$g7OcPZb_G)k@62Lf5X+EiE~Hwt2X!>k^Z&PE2sR8syp4~XQb_K`5TZwA^8{X`xU?2zv53&|J%eq^1o4jh2u{+euUHisd-rcivK|Q4Jz+H<u~N|@0LH%;s<o~|5N`<^}DEePI`apzp?uN#((3uH~&h%U)N8q_4B=c``n)EuQUC97yo~=5B1BLet%bgpX%Q$ZX4CFcl!OQem&LirTYI$zt`Fu{a&SCPxX5#KVVn?pX%4A`n^uSpYj9hzQ3#gpX$$Aet<bYIQ@R7f28cGei7CGSN@lY{rP$O`v1uP<MjW#`u$x0oACp*{<pEW`rq>Xq5d`0?>G9_I{v`))17|3(@$6NldgV!SN~k<4L9}cQ~h{3UpDpQyZZN%yVKOauit~}$E*JA^vgH&^XL6O)Sp-VZ$p0`>Ax@L9i(6HxFJq|Tj|HU@#pl{slDEJGW~JK%}Vvt{r%_q-y8bf(fUpGyPf{F($A*)(^h}Gsh{1@zmD{8seW`r|2fi6Rr<+Pf7t0iJN;qjFLnCEihJDEe@^v>o&K|{-yZJ2sXt8hpIqFcq2C<oM>+lBu6{DrZ*uy-uKw>aFMoVh4rHpI?DUrd{p(0S3g{;<^>dW|aj^a{{baA-ThGmNJ=suyInW<o>o>#auXXN${_s-&8R+NE?VI|;UHxZOZ%*}}UB8R<hlLI>)&HgXmsUSm?_ZYs!LI(^=?7E&SfyLEt3RCT2TL8;P`@_S-*xrILHo_`zpK9s>+h)!EVd8zZ&UrpK>v1X@APNg{)y_}s{Jq3zfJXP1N~N{UrY51t8QefAFK3Vss5<jH&5#Kq3-uuztC}8sQ#qW4NmnZ-Sb=hxvqX3)Gt}+k45?;RzFV8e^5Wp=)Xz*xKRHs)h}`SZLxk_s6SP8t5o~R{fE{MJ9NVi9kEnDEYg3I`f;KDTdKd}^xI<nxKMw^>aRilw4wbr_0y<+i;P=B{VuG3mg{d#b;PiK*?8`n+GG7O^ZZhOEYuHM=(bJmv3^&sAJx*|O2@C$&!YNW3%&9^{VllvUhKPD`dNJZuJyCr`UUi(TKZQ~hj;4#Tj>8Z^;>fNgt`6;)nBRg7E+HS)sKPtEvkN(>bE%knLxiK(hsTiVw3wX>ra*b2-Od9`UzCOpypxk>KD-b*D>#Uc>ZPn>sbFD>JJ$G15kgy)IZ4U*XKF{sr~`2A9MBU|4RR$t6yN-Pv{o}_5V}-fmA<0=|(tvTK`S!pLN~0sDFm_$9erStAEw?b>^>O{cKRbI*q@i^{aKCJJj(Odsjc|#xJOUY}H@>>-s~NKP&4G&HjnjAJY28^!~Q~@E_F=D*vagALRKT%lf~x{t(xH3Eh!y{hq4-bM-H>{two_$ojix{oK<}Eb0KC?Jw%r@cv8GfARXQb^VO5zhLz{uKq&RpV0abSN}rquj&xe`j?IRjVOOt)la1LC$9d&&A%#t-RDoU{C}E%-_5_(`?p^HdzwG(^51Fx`f%TV{`xeJwVS_A^S>u~<I}uen*W{Vzq@&3Bma7uKkf1#)BHb`$C&1?(fmV|KbPhoy8OjO{w&Jh>*g=I_21>MrTJ$v{|);8)j3Arf1UsH&+><w`8)3ZIRB@ce-rxO-Fw&k1rR@8`mbI5-NzqI{N294iNC}6`4E5KjGxo^g^PcCeu#-*!}#+e{@RK^)A)ycejGnC@z*eZI*C6{{W>^)3ge%PeBZtJCyZYL?tB!#<llcCzohfW@cv=^aZxYQia*lvuXSfcJSU7_<o@qCelU$+r16K*|1IJ$Vf+EaFP6G{UH|**yd>%?jsJYte@*?z*8jTeKW+Ly4?Crcf&HJm{!3^t^Sq7!&H5jk{zH0y=YQJu|2TVhe%w9(IeV~wGy5O0e+~98)BS6>|E%kmJI}cNXOcf>_HTCoXZAn6{TJLn4d>TiJICRy=T}hoH+_G#es|a3bp4dNPjLQq+^O#Ty*qy%^QMXe9rNe^KFaj&'


def _decode_int8_table(s: str, shape: Tuple[int, ...]) -> torch.Tensor:
    raw = zlib.decompress(base64.b85decode(s.encode("ascii")))
    # bytearray gives a writable buffer and avoids the warning emitted by
    # torch.frombuffer on immutable bytes.
    t = torch.frombuffer(bytearray(raw), dtype=torch.int8).clone()
    return t.reshape(shape)


_C4_CPU = _decode_int8_table(_C_DATA, (135, 8, 8))
_A2x2_CPU = _decode_int8_table(_A2_DATA, (270, 8, 8))
_A3x4_CPU = _decode_int8_table(_A3_DATA, (2240, 8, 8))


# -----------------------------------------------------------------------------
# Basic E8 CVP
# -----------------------------------------------------------------------------


def _make_e8_basis(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """8x8 column basis for D8^+ = E8, determinant ±1."""
    B = torch.zeros((8, 8), device=device, dtype=dtype)
    B[:, 0] = 0.5
    for col, i in enumerate(range(1, 7), start=1):
        B[i, col] = 1.0
        B[i + 1, col] = -1.0
    B[6, 7] = 1.0
    B[7, 7] = 1.0
    return B


@torch.no_grad()
def d8_cvp(x: torch.Tensor) -> torch.Tensor:
    """Exact batched CVP onto D8 = {z∈Z^8 : sum z_i even}."""
    if x.shape[-1] != 8:
        raise ValueError("d8_cvp expects x[...,8]")
    if not x.dtype.is_floating_point:
        x = x.to(torch.get_default_dtype())

    r = torch.round(x)
    odd = (r.to(torch.int64).sum(dim=-1) & 1).to(x.dtype)
    frac = x - r

    # If parity is odd, change the rounded coordinate closest to its *other*
    # adjacent integer.  The penalty is 1 - 2|frac_i|, hence choose max |frac|.
    j = frac.abs().argmax(dim=-1, keepdim=True)
    fj = frac.gather(-1, j)
    direction = torch.where(fj >= 0, torch.ones_like(fj), -torch.ones_like(fj))
    correction = torch.zeros_like(x)
    correction.scatter_(-1, j, direction * odd.unsqueeze(-1))
    return r + correction


@torch.no_grad()
def e8_cvp(x: torch.Tensor) -> torch.Tensor:
    """Exact batched CVP onto E8 = D8 ∪ ((1/2,...,1/2)+D8)."""
    if x.shape[-1] != 8:
        raise ValueError("e8_cvp expects x[...,8]")
    if not x.dtype.is_floating_point:
        x = x.to(torch.get_default_dtype())
    h = torch.tensor(0.5, device=x.device, dtype=x.dtype)
    y0 = d8_cvp(x)
    y1 = h + d8_cvp(x - h)
    d0 = (x - y0).square().sum(dim=-1)
    d1 = (x - y1).square().sum(dim=-1)
    return torch.where((d1 < d0).unsqueeze(-1), y1, y0)


@torch.no_grad()
def nearest_e8_mod_2(x: torch.Tensor, rep: torch.Tensor) -> torch.Tensor:
    """Nearest point to x in rep + 2E8; x and rep broadcast to (...,8)."""
    return rep + 2.0 * e8_cvp((x - rep) * 0.5)


# -----------------------------------------------------------------------------
# Compact W(D8) subset-DP plan
# -----------------------------------------------------------------------------


@dataclass
class _WD8Stage:
    prev: torch.Tensor   # (D_cur, K)
    col: torch.Tensor    # (D_cur, K)
    sign: torch.Tensor   # (D_cur, K)


def _make_wd8_plan(device: torch.device, dtype: torch.dtype):
    stages = []
    prev_masks = [0]
    for i in range(8):
        cur_masks = [m for m in range(256) if m.bit_count() == i + 1]
        prev_pos = {m: k for k, m in enumerate(prev_masks)}
        dcur = 2 * len(cur_masks)
        kcount = 2 * (i + 1)
        prev = torch.empty((dcur, kcount), dtype=torch.long)
        col = torch.empty((dcur, kcount), dtype=torch.long)
        sign = torch.empty((dcur, kcount), dtype=dtype)

        for mi, mask in enumerate(cur_masks):
            bits = [j for j in range(8) if (mask >> j) & 1]
            for pnew in (0, 1):
                dst = 2 * mi + pnew
                q = 0
                for j in bits:
                    pmask = mask ^ (1 << j)
                    pi = prev_pos[pmask]
                    # Positive sign: parity unchanged.
                    prev[dst, q] = 2 * pi + pnew
                    col[dst, q] = j
                    sign[dst, q] = 1.0
                    q += 1
                    # Negative sign: parity toggles.
                    prev[dst, q] = 2 * pi + (pnew ^ 1)
                    col[dst, q] = j
                    sign[dst, q] = -1.0
                    q += 1

        stages.append(_WD8Stage(prev.to(device), col.to(device), sign.to(device)))
        prev_masks = cur_masks
    return stages


@dataclass
class _B8Stage:
    prev: torch.Tensor
    col: torch.Tensor


def _make_b8_plan(device: torch.device):
    """Subset-DP plan for the full signed-permutation group B8."""
    stages = []
    prev_masks = [0]
    for i in range(8):
        cur_masks = [m for m in range(256) if m.bit_count() == i + 1]
        prev_pos = {m: k for k, m in enumerate(prev_masks)}
        prev = torch.empty((len(cur_masks), i + 1), dtype=torch.long)
        col = torch.empty_like(prev)
        for mi, mask in enumerate(cur_masks):
            bits = [j for j in range(8) if (mask >> j) & 1]
            for q, j in enumerate(bits):
                prev[mi, q] = prev_pos[mask ^ (1 << j)]
                col[mi, q] = j
        stages.append(_B8Stage(prev.to(device), col.to(device)))
        prev_masks = cur_masks
    return stages


# -----------------------------------------------------------------------------
# Main projector
# -----------------------------------------------------------------------------


@dataclass
class CVPInfo:
    squared_error: torch.Tensor
    outer_coset: torch.Tensor


class E8MatrixLattice:
    """
    Fast batched projector for E8⊗E8 and its m=1,2,3,4 scaled-orthogonal shells.

    Parameters
    ----------
    device:
        CPU or CUDA device.
    dtype:
        torch.float32 for speed or torch.float64 for more robust tie decisions.
    preload_m2:
        If True, materialize the 36,450 m=2 W(D8)-coset transforms immediately.
        If False, they are built lazily on the first project_m2() call.
    preload_m3:
        If True, materialize the 302,400 m=3 W(D8)-coset transforms immediately.
        This is mainly useful for CPU reference/testing; the Triton handoff can
        preload it on GPU as well (about 77 MB in float32).
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        *,
        preload_m2: bool = False,
        preload_m3: bool = False,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")

        self.B = _make_e8_basis(self.device, self.dtype)
        self.Binv = torch.linalg.inv(self.B)

        # E8/2E8 representatives B*b, b∈{0,1}^8.
        ids = torch.arange(256, device=self.device, dtype=torch.long)
        bits = ((ids[:, None] >> torch.arange(8, device=self.device)) & 1).to(self.dtype)
        self.e8_mod2_reps = bits @ self.B.T
        self.xor = torch.bitwise_xor(ids[:, None], ids[None, :])

        # Signed-permutation DP plans.
        self._wd8_plan = _make_wd8_plan(self.device, self.dtype)
        self._b8_plan = _make_b8_plan(self.device)

        # Fast E8/2E8 branch-metric tables.  A residue class is determined by
        # (a) integer/half-integer type a, (b) an even coordinate-parity
        # pattern modulo complement, and (c) sum(k)/2 mod 2.
        patterns = []
        for mask in range(256):
            bits8 = [(mask >> i) & 1 for i in range(8)]
            if bits8[0] == 0 and (sum(bits8) & 1) == 0:
                patterns.append(bits8)
        self._mod2_patterns_i = torch.tensor(patterns, device=self.device, dtype=torch.long)
        self._mod2_patterns = self._mod2_patterns_i.to(self.dtype)
        lookup = {tuple(row): i for i, row in enumerate(patterns)}
        id_map = torch.empty((2, 64, 2), dtype=torch.long)
        for rid, yy in enumerate(self.e8_mod2_reps.detach().cpu()):
            two = torch.round(2.0 * yy).to(torch.long)
            a = int(two[0].item()) & 1
            kk = torch.round(yy - 0.5 * a).to(torch.long)
            pp = kk & 1
            if int(pp[0].item()):
                pp = 1 - pp
            ci = lookup[tuple(int(v) for v in pp.tolist())]
            ss = (int(kk.sum().item()) % 4) // 2
            id_map[a, ci, ss] = rid
        self._mod2_id_map = id_map.to(self.device)
        self._residue_powers = (1 << torch.arange(8, device=self.device, dtype=torch.long))

        # 135 reps C_k for W(E8)/W(D8).
        self.C1 = (_C4_CPU.to(self.device, dtype=self.dtype) * 0.25).contiguous()

        # 270 m=2 image reps A_a.
        self.A2 = (_A2x2_CPU.to(self.device, dtype=self.dtype) * 0.5).contiguous()
        self._T2: Optional[torch.Tensor] = None
        self._T2_flatcols: Optional[torch.Tensor] = None
        if preload_m2:
            self._ensure_T2()

        # 2240 m=3 image reps A_a.  Stored as quarter-integers.
        self.A3 = (_A3x4_CPU.to(self.device, dtype=self.dtype) * 0.25).contiguous()
        self._T3: Optional[torch.Tensor] = None
        self._T3_flatcols: Optional[torch.Tensor] = None
        if preload_m3:
            self._ensure_T3()

    # ------------------------- input helpers -------------------------

    def _as_matrix_batch(self, x: torch.Tensor):
        if x.ndim >= 2 and x.shape[-2:] == (8, 8):
            lead = x.shape[:-2]
            matrix_input = True
        elif x.ndim >= 1 and x.shape[-1] == 64:
            lead = x.shape[:-1]
            matrix_input = False
        else:
            raise ValueError("expected x with shape (...,8,8) or (...,64)")
        X = x.to(device=self.device, dtype=self.dtype).reshape(-1, 8, 8)
        return X, lead, matrix_input

    @staticmethod
    def _restore_matrix_shape(Y: torch.Tensor, lead, matrix_input: bool):
        if matrix_input:
            return Y.reshape(*lead, 8, 8)
        return Y.reshape(*lead, 64)

    # ------------------------- W(D8) kernel -------------------------

    @torch.no_grad()
    def _wd8_scores(self, Y: torch.Tensor) -> torch.Tensor:
        """max_{D∈W(D8)} <Y,D>, vectorized over Y[...,8,8]."""
        lead = Y.shape[:-2]
        Yf = Y.reshape(-1, 8, 8)
        n = Yf.shape[0]
        dp = torch.full((n, 2), float("-inf"), device=self.device, dtype=self.dtype)
        dp[:, 0] = 0.0
        for i, st in enumerate(self._wd8_plan):
            cand = dp[:, st.prev] + Yf[:, i, st.col] * st.sign
            dp = cand.amax(dim=-1)
        # Final active masks list contains only mask=255; parity-even is state 0.
        return dp[:, 0].reshape(lead)

    @torch.no_grad()
    def _wd8_best(self, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return maximizing D∈W(D8) and its score."""
        lead = Y.shape[:-2]
        Yf = Y.reshape(-1, 8, 8)
        n = Yf.shape[0]
        dp = torch.full((n, 2), float("-inf"), device=self.device, dtype=self.dtype)
        dp[:, 0] = 0.0
        args = []

        for i, st in enumerate(self._wd8_plan):
            cand = dp[:, st.prev] + Yf[:, i, st.col] * st.sign
            dp, a = cand.max(dim=-1)
            args.append(a.to(torch.uint8))

        score = dp[:, 0]
        state = torch.zeros(n, dtype=torch.long, device=self.device)
        cols = torch.empty((n, 8), dtype=torch.long, device=self.device)
        signs = torch.empty((n, 8), dtype=self.dtype, device=self.device)
        ar = torch.arange(n, device=self.device)

        for i in range(7, -1, -1):
            st = self._wd8_plan[i]
            a = args[i][ar, state].to(torch.long)
            cols[:, i] = st.col[state, a]
            signs[:, i] = st.sign[state, a]
            state = st.prev[state, a]

        D = torch.zeros((n, 8, 8), dtype=self.dtype, device=self.device)
        D[ar[:, None], torch.arange(8, device=self.device)[None, :], cols] = signs
        return D.reshape(*lead, 8, 8), score.reshape(lead)

    @torch.no_grad()
    def _b8_scores(self, Y: torch.Tensor) -> torch.Tensor:
        """Upper bound max over all signed permutations B8."""
        lead = Y.shape[:-2]
        Yf = Y.reshape(-1, 8, 8).abs()
        dp = torch.zeros((Yf.shape[0], 1), device=self.device, dtype=self.dtype)
        for i, st in enumerate(self._b8_plan):
            dp = (dp[:, st.prev] + Yf[:, i, st.col]).amax(dim=-1)
        return dp[:, 0].reshape(lead)

    @torch.no_grad()
    def _e8_mod2_costs_fast(self, t: torch.Tensor) -> torch.Tensor:
        """
        Exact squared distances from t[...,8] to all 256 cosets of 2E8.

        This uses the concrete description
          2E8 = {z in Z^8: all z_i have one parity, sum(z)=0 mod 4}.
        It is substantially faster than invoking E8 CVP independently for all
        256 residue classes.
        """
        lead = t.shape[:-1]
        x = t.reshape(-1, 8)
        n = x.shape[0]
        out = torch.empty((n, 256), device=self.device, dtype=self.dtype)
        P = self._mod2_patterns
        Pb = self._mod2_patterns_i.bool()

        for a in (0, 1):
            r = x - 0.5 * float(a)
            ve = 2.0 * torch.round(r * 0.5)
            vo = 2.0 * torch.round((r - 1.0) * 0.5) + 1.0
            ce = (r - ve).square()
            co = (r - vo).square()
            pe = torch.minimum((r - (ve + 2.0)).square(), (r - (ve - 2.0)).square()) - ce
            po = torch.minimum((r - (vo + 2.0)).square(), (r - (vo - 2.0)).square()) - co

            # Canonical parity pattern q=P.
            base = ce.sum(-1, keepdim=True) + (co - ce) @ P.T
            vsum = ve.sum(-1, keepdim=True) + (vo - ve) @ P.T
            penalty = torch.where(Pb[None, :, :], po[:, None, :], pe[:, None, :]).amin(-1)
            sb = ((torch.round(vsum).to(torch.long) // 2) & 1)
            z = torch.zeros_like(base)
            c0 = base + torch.where(sb == 0, z, penalty)
            c1 = base + torch.where(sb == 1, z, penalty)

            # Complement parity pattern 1-P represents the same parity class.
            base2 = co.sum(-1, keepdim=True) + (ce - co) @ P.T
            vsum2 = vo.sum(-1, keepdim=True) + (ve - vo) @ P.T
            penalty2 = torch.where(Pb[None, :, :], pe[:, None, :], po[:, None, :]).amin(-1)
            sb2 = ((torch.round(vsum2).to(torch.long) // 2) & 1)
            d0 = base2 + torch.where(sb2 == 0, z, penalty2)
            d1 = base2 + torch.where(sb2 == 1, z, penalty2)

            cc = torch.stack((torch.minimum(c0, d0), torch.minimum(c1, d1)), dim=-1)
            out[:, self._mod2_id_map[a].reshape(-1)] = cc.reshape(n, -1)

        return out.reshape(*lead, 256)

    def _tol(self, scale: Optional[torch.Tensor] = None):
        base = 2e-5 if self.dtype == torch.float32 else 2e-11
        if scale is None:
            return base
        return base * (1.0 + scale.abs())

    # ------------------------- shell transforms -------------------------

    @torch.no_grad()
    def _ensure_T2(self) -> torch.Tensor:
        if self._T2 is None:
            # T_{a,k}=A_a C_k; S_2 = ⋃ T_{a,k} W(D8).
            self._T2 = torch.einsum("aij,kjl->akil", self.A2, self.C1).reshape(-1, 8, 8).contiguous()
        if self._T2_flatcols is None:
            # Columns of all T_t packed as an 8 x (N*8) matrix.  This lets
            # the cheap m=2 upper bound use eight efficient GEMMs.
            self._T2_flatcols = self._T2.permute(1, 0, 2).reshape(8, -1).contiguous()
        return self._T2

    @torch.no_grad()
    def _ensure_T3(self) -> torch.Tensor:
        if self._T3 is None:
            # T_{a,k}=A_a C_k; S_3 = ⋃ T_{a,k} W(D8).
            self._T3 = torch.einsum("aij,kjl->akil", self.A3, self.C1).reshape(-1, 8, 8).contiguous()
        if self._T3_flatcols is None:
            self._T3_flatcols = self._T3.permute(1, 0, 2).reshape(8, -1).contiguous()
        return self._T3

    @torch.no_grad()
    def _project_transform_shell_flat(
        self,
        X: torch.Tensor,
        transforms: torch.Tensor,
        *,
        norm2: float,
        transform_chunk: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project X[B,8,8] onto ⋃_t transforms[t] W(D8)."""
        bsz = X.shape[0]
        nt = transforms.shape[0]
        best_score = torch.full((bsz,), float("-inf"), device=self.device, dtype=self.dtype)
        best_t = torch.zeros((bsz,), dtype=torch.long, device=self.device)

        for lo in range(0, nt, transform_chunk):
            T = transforms[lo: lo + transform_chunk]
            # Y[b,t] = T_t^T X_b.
            Y = torch.einsum("tji,bjk->btik", T, X)
            s = self._wd8_scores(Y).reshape(bsz, -1)
            cs, ci = s.max(dim=1)
            improve = cs > best_score
            best_score = torch.where(improve, cs, best_score)
            best_t = torch.where(improve, lo + ci, best_t)

        Tbest = transforms[best_t]
        Ybest = torch.einsum("bji,bjk->bik", Tbest, X)
        Dbest, _ = self._wd8_best(Ybest)
        M = Tbest @ Dbest
        d2 = X.square().sum(dim=(-2, -1)) + norm2 - 2.0 * best_score
        return M, torch.clamp_min(d2, 0.0)

    @torch.no_grad()
    def _project_m1_branch_bound_flat(
        self,
        X: torch.Tensor,
        *,
        initial_candidates: int = 8,
        pair_chunk_size: int = 4096,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact CPU-friendly branch-and-bound projection onto S_1."""
        T = self.C1
        nt = T.shape[0]
        TF = T.permute(1, 0, 2).reshape(8, -1).contiguous()
        bsz = X.shape[0]
        k0 = min(int(initial_candidates), nt)
        rowmax = torch.zeros((bsz, nt, 8), device=self.device, dtype=self.dtype)
        colsum = torch.zeros((bsz, nt), device=self.device, dtype=self.dtype)
        for k in range(8):
            z = (X[:, :, k] @ TF).reshape(bsz, nt, 8).abs()
            rowmax = torch.maximum(rowmax, z)
            colsum += z.amax(dim=-1)
        ub = torch.minimum(rowmax.sum(dim=-1), colsum)
        del rowmax, colsum

        _, topi = torch.topk(ub, k0, dim=-1, largest=True, sorted=False)
        Y0 = torch.matmul(T[topi].transpose(-1, -2), X[:, None, :, :])
        s0 = self._wd8_scores(Y0).reshape(bsz, k0)
        best_score, kk = s0.max(dim=-1)
        ar = torch.arange(bsz, device=self.device)
        best_t = topi[ar, kk]

        seeded = torch.zeros((bsz, nt), device=self.device, dtype=torch.bool)
        seeded.scatter_(1, topi, True)
        pairs = ((ub + self._tol(best_score)[:, None] > best_score[:, None]) & (~seeded)).nonzero(as_tuple=False)
        if pairs.numel():
            pb, pt = pairs[:, 0], pairs[:, 1]
            YY = torch.bmm(T[pt].transpose(1, 2), X[pb])
            bu = self._b8_scores(YY)
            keep = bu + self._tol(best_score[pb]) > best_score[pb]
            pb, pt = pb[keep], pt[keep]
            for lo in range(0, pb.numel(), pair_chunk_size):
                bb = pb[lo:lo + pair_chunk_size]
                tt = pt[lo:lo + pair_chunk_size]
                YY = torch.bmm(T[tt].transpose(1, 2), X[bb])
                ss = self._wd8_scores(YY)
                for q in range(ss.numel()):
                    b = int(bb[q].item())
                    if ss[q] > best_score[b]:
                        best_score[b] = ss[q]
                        best_t[b] = tt[q]

        Tbest = T[best_t]
        Ybest = torch.bmm(Tbest.transpose(1, 2), X)
        Dbest, _ = self._wd8_best(Ybest)
        M = torch.bmm(Tbest, Dbest)
        d2 = X.square().sum(dim=(-2, -1)) + 8.0 - 2.0 * best_score
        return M, torch.clamp_min(d2, 0.0)

    @torch.no_grad()
    def _project_m2_branch_bound_flat(
        self,
        X: torch.Tensor,
        *,
        initial_candidates: int = 64,
        pair_chunk_size: int = 4096,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact CPU-friendly branch-and-bound projection onto S_2."""
        T = self._ensure_T2()
        TF = self._T2_flatcols
        nt = T.shape[0]
        bsz = X.shape[0]
        k0 = min(int(initial_candidates), nt)

        # Upper bound for a W(D8) score: any signed permutation picks exactly
        # one entry from each row and column, so both row-max and column-max
        # sums are valid bounds.
        rowmax = torch.zeros((bsz, nt, 8), device=self.device, dtype=self.dtype)
        colsum = torch.zeros((bsz, nt), device=self.device, dtype=self.dtype)
        for k in range(8):
            z = (X[:, :, k] @ TF).reshape(bsz, nt, 8).abs()
            rowmax = torch.maximum(rowmax, z)
            colsum += z.amax(dim=-1)
        ub = torch.minimum(rowmax.sum(dim=-1), colsum)
        del rowmax, colsum

        # Seed with exact scores for the largest cheap upper bounds.
        _, topi = torch.topk(ub, k0, dim=-1, largest=True, sorted=False)
        Y0 = torch.matmul(T[topi].transpose(-1, -2), X[:, None, :, :])
        s0 = self._wd8_scores(Y0).reshape(bsz, k0)
        best_score, kk = s0.max(dim=-1)
        ar = torch.arange(bsz, device=self.device)
        best_t = topi[ar, kk]

        seeded = torch.zeros((bsz, nt), device=self.device, dtype=torch.bool)
        seeded.scatter_(1, topi, True)
        tol = self._tol(best_score)[:, None]
        pairs = ((ub + tol > best_score[:, None]) & (~seeded)).nonzero(as_tuple=False)
        del ub, seeded, Y0, s0

        if pairs.numel():
            pb, pt = pairs[:, 0], pairs[:, 1]
            keep_b, keep_t = [], []
            # Tighter upper bound: relax even-sign W(D8) to all signed
            # permutations B8.  This DP is 4x smaller than the W(D8) DP.
            for lo in range(0, pb.numel(), pair_chunk_size):
                bb = pb[lo:lo + pair_chunk_size]
                tt = pt[lo:lo + pair_chunk_size]
                YY = torch.bmm(T[tt].transpose(1, 2), X[bb])
                bu = self._b8_scores(YY)
                keep = bu + self._tol(best_score[bb]) > best_score[bb]
                if keep.any():
                    keep_b.append(bb[keep])
                    keep_t.append(tt[keep])

            if keep_b:
                kb = torch.cat(keep_b)
                kt = torch.cat(keep_t)
                for lo in range(0, kb.numel(), pair_chunk_size):
                    bb = kb[lo:lo + pair_chunk_size]
                    tt = kt[lo:lo + pair_chunk_size]
                    YY = torch.bmm(T[tt].transpose(1, 2), X[bb])
                    ss = self._wd8_scores(YY)
                    # The survivor count is small; grouped Python updates avoid
                    # materializing a dense [B,N] score array.
                    for q in range(ss.numel()):
                        b = int(bb[q].item())
                        if ss[q] > best_score[b]:
                            best_score[b] = ss[q]
                            best_t[b] = tt[q]

        Tbest = T[best_t]
        Ybest = torch.bmm(Tbest.transpose(1, 2), X)
        Dbest, _ = self._wd8_best(Ybest)
        M = torch.bmm(Tbest, Dbest)
        d2 = X.square().sum(dim=(-2, -1)) + 16.0 - 2.0 * best_score
        return M, torch.clamp_min(d2, 0.0)

    @torch.no_grad()
    def _project_m3_branch_bound_flat(
        self,
        X: torch.Tensor,
        *,
        initial_candidates: int = 128,
        pair_chunk_size: int = 4096,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact CPU reference branch-and-bound projection onto S_3.

        This materializes the 302,400 W(D8)-coset transforms, so it is intended
        primarily as a correctness oracle for the CUDA/Triton implementation,
        not as the final high-throughput m=3 path.
        """
        T = self._ensure_T3()
        TF = self._T3_flatcols
        nt = T.shape[0]
        bsz = X.shape[0]
        k0 = min(int(initial_candidates), nt)

        rowmax = torch.zeros((bsz, nt, 8), device=self.device, dtype=self.dtype)
        colsum = torch.zeros((bsz, nt), device=self.device, dtype=self.dtype)
        for k in range(8):
            z = (X[:, :, k] @ TF).reshape(bsz, nt, 8).abs()
            rowmax = torch.maximum(rowmax, z)
            colsum += z.amax(dim=-1)
        ub = torch.minimum(rowmax.sum(dim=-1), colsum)
        del rowmax, colsum

        _, topi = torch.topk(ub, k0, dim=-1, largest=True, sorted=False)
        Y0 = torch.matmul(T[topi].transpose(-1, -2), X[:, None, :, :])
        s0 = self._wd8_scores(Y0).reshape(bsz, k0)
        best_score, kk = s0.max(dim=-1)
        ar = torch.arange(bsz, device=self.device)
        best_t = topi[ar, kk]

        seeded = torch.zeros((bsz, nt), device=self.device, dtype=torch.bool)
        seeded.scatter_(1, topi, True)
        pairs = ((ub + self._tol(best_score)[:, None] > best_score[:, None]) & (~seeded)).nonzero(as_tuple=False)
        del ub, seeded, Y0, s0

        if pairs.numel():
            pb, pt = pairs[:, 0], pairs[:, 1]
            keep_b, keep_t = [], []
            for lo in range(0, pb.numel(), pair_chunk_size):
                bb = pb[lo:lo + pair_chunk_size]
                tt = pt[lo:lo + pair_chunk_size]
                YY = torch.bmm(T[tt].transpose(1, 2), X[bb])
                bu = self._b8_scores(YY)
                keep = bu + self._tol(best_score[bb]) > best_score[bb]
                if keep.any():
                    keep_b.append(bb[keep])
                    keep_t.append(tt[keep])

            if keep_b:
                kb = torch.cat(keep_b)
                kt = torch.cat(keep_t)
                for lo in range(0, kb.numel(), pair_chunk_size):
                    bb = kb[lo:lo + pair_chunk_size]
                    tt = kt[lo:lo + pair_chunk_size]
                    YY = torch.bmm(T[tt].transpose(1, 2), X[bb])
                    ss = self._wd8_scores(YY)
                    for q in range(ss.numel()):
                        b = int(bb[q].item())
                        if ss[q] > best_score[b]:
                            best_score[b] = ss[q]
                            best_t[b] = tt[q]

        Tbest = T[best_t]
        Ybest = torch.bmm(Tbest.transpose(1, 2), X)
        Dbest, _ = self._wd8_best(Ybest)
        M = torch.bmm(Tbest, Dbest)
        d2 = X.square().sum(dim=(-2, -1)) + 24.0 - 2.0 * best_score
        return M, torch.clamp_min(d2, 0.0)

    @torch.no_grad()
    def _project_shell(
        self,
        x: torch.Tensor,
        *,
        m: int,
        batch_chunk_size: Optional[int],
        transform_chunk: Optional[int],
        return_distance: bool,
    ):
        X, lead, matrix_input = self._as_matrix_batch(x)
        if m == 1:
            T = self.C1
            norm2 = 8.0
            if transform_chunk is None:
                transform_chunk = 135
        elif m == 2:
            T = self._ensure_T2()
            norm2 = 16.0
            if transform_chunk is None:
                transform_chunk = 256
        elif m == 3:
            T = self._ensure_T3()
            norm2 = 24.0
            if transform_chunk is None:
                transform_chunk = 128
        else:
            raise ValueError("this package implements shell m=1, m=2, or m=3")

        n = X.shape[0]
        if batch_chunk_size is None:
            batch_chunk_size = n
        if batch_chunk_size <= 0 or transform_chunk <= 0:
            raise ValueError("chunk sizes must be positive")

        ys, ds = [], []
        for lo in range(0, n, batch_chunk_size):
            Y, d2 = self._project_transform_shell_flat(
                X[lo:lo + batch_chunk_size], T,
                norm2=norm2,
                transform_chunk=transform_chunk,
            )
            ys.append(Y)
            ds.append(d2)

        Y = torch.cat(ys, dim=0)
        d2 = torch.cat(ds, dim=0)
        Y = self._restore_matrix_shape(Y, lead, matrix_input)
        if return_distance:
            return Y, d2.reshape(lead)
        return Y

    @torch.no_grad()
    def project_m1(
        self,
        x: torch.Tensor,
        *,
        batch_chunk_size: Optional[int] = None,
        transform_chunk: Optional[int] = None,
        return_distance: bool = False,
        algorithm: str = "auto",
        initial_candidates: int = 8,
        pair_chunk_size: int = 4096,
    ):
        """Exact projection onto S_1={M∈E8⊗E8:M^T M=I}."""
        if algorithm == "auto":
            algorithm = "branch_bound" if self.device.type == "cpu" else "exhaustive"
        if algorithm == "exhaustive":
            return self._project_shell(
                x, m=1, batch_chunk_size=batch_chunk_size,
                transform_chunk=transform_chunk, return_distance=return_distance,
            )
        if algorithm != "branch_bound":
            raise ValueError("algorithm must be 'auto', 'branch_bound', or 'exhaustive'")
        X, lead, matrix_input = self._as_matrix_batch(x)
        n = X.shape[0]
        if batch_chunk_size is None:
            batch_chunk_size = 128 if self.device.type == "cpu" else n
        ys, ds = [], []
        for lo in range(0, n, batch_chunk_size):
            yy, dd = self._project_m1_branch_bound_flat(
                X[lo:lo + batch_chunk_size],
                initial_candidates=initial_candidates,
                pair_chunk_size=pair_chunk_size,
            )
            ys.append(yy); ds.append(dd)
        Y = torch.cat(ys, dim=0)
        d2 = torch.cat(ds, dim=0)
        Y = self._restore_matrix_shape(Y, lead, matrix_input)
        if return_distance:
            return Y, d2.reshape(lead)
        return Y

    @torch.no_grad()
    def project_m2(
        self,
        x: torch.Tensor,
        *,
        batch_chunk_size: Optional[int] = None,
        transform_chunk: Optional[int] = None,
        return_distance: bool = False,
        algorithm: str = "auto",
        initial_candidates: int = 64,
        pair_chunk_size: int = 4096,
    ):
        """
        Exact projection onto S_2={M∈E8⊗E8:M^T M=2I}.

        algorithm='branch_bound' is the new CPU-oriented exact algorithm.
        algorithm='exhaustive' retains the old 36,450-coset scan.
        'auto' uses branch-and-bound on CPU and exhaustive search on CUDA.
        """
        if algorithm == "auto":
            algorithm = "branch_bound" if self.device.type == "cpu" else "exhaustive"
        if algorithm == "exhaustive":
            return self._project_shell(
                x, m=2, batch_chunk_size=batch_chunk_size,
                transform_chunk=transform_chunk, return_distance=return_distance,
            )
        if algorithm != "branch_bound":
            raise ValueError("algorithm must be 'auto', 'branch_bound', or 'exhaustive'")

        X, lead, matrix_input = self._as_matrix_batch(x)
        n = X.shape[0]
        if batch_chunk_size is None:
            batch_chunk_size = 32 if self.device.type == "cpu" else n
        ys, ds = [], []
        for lo in range(0, n, batch_chunk_size):
            yy, dd = self._project_m2_branch_bound_flat(
                X[lo:lo + batch_chunk_size],
                initial_candidates=initial_candidates,
                pair_chunk_size=pair_chunk_size,
            )
            ys.append(yy); ds.append(dd)
        Y = torch.cat(ys, dim=0)
        d2 = torch.cat(ds, dim=0)
        Y = self._restore_matrix_shape(Y, lead, matrix_input)
        if return_distance:
            return Y, d2.reshape(lead)
        return Y

    @torch.no_grad()
    def project_m3(
        self,
        x: torch.Tensor,
        *,
        batch_chunk_size: Optional[int] = None,
        transform_chunk: Optional[int] = None,
        return_distance: bool = False,
        algorithm: str = "branch_bound",
        initial_candidates: int = 128,
        pair_chunk_size: int = 4096,
    ):
        """Exact CPU reference projection onto S_3={M:M^T M=3I}.

        The branch-bound path is meant as a reference/oracle.  The handoff
        package adds a Triton CUDA implementation in e8_triton_projector.py.
        """
        if algorithm == "auto":
            algorithm = "branch_bound"
        if algorithm == "exhaustive":
            return self._project_shell(
                x, m=3, batch_chunk_size=batch_chunk_size,
                transform_chunk=transform_chunk, return_distance=return_distance,
            )
        if algorithm != "branch_bound":
            raise ValueError("algorithm must be 'auto', 'branch_bound', or 'exhaustive'")
        X, lead, matrix_input = self._as_matrix_batch(x)
        n = X.shape[0]
        if batch_chunk_size is None:
            batch_chunk_size = 1 if self.device.type == "cpu" else n
        ys, ds = [], []
        for lo in range(0, n, batch_chunk_size):
            yy, dd = self._project_m3_branch_bound_flat(
                X[lo:lo + batch_chunk_size],
                initial_candidates=initial_candidates,
                pair_chunk_size=pair_chunk_size,
            )
            ys.append(yy); ds.append(dd)
        Y = torch.cat(ys, dim=0)
        d2 = torch.cat(ds, dim=0)
        Y = self._restore_matrix_shape(Y, lead, matrix_input)
        if return_distance:
            return Y, d2.reshape(lead)
        return Y

    @torch.no_grad()
    def project_m4(
        self,
        x: torch.Tensor,
        *,
        batch_chunk_size: Optional[int] = None,
        outer_a_chunk: int = 2,
        inner_initial_candidates: int = 64,
        inner_pair_chunk_size: int = 4096,
        return_distance: bool = False,
    ):
        """Exact projection onto S_4 using S_4 = S_2 S_2.

        Completeness of the product description was checked independently: the
        270^2 products of multiplier-2 image representatives yield exactly
        36,766 distinct multiplier-4 image sublattices, matching the E8
        similar-sublattice counting formula.  Duplicates are harmless here, so
        runtime uses the simple 270-by-S_2 recursive search rather than storing
        all 36,766 image classes.

        This is primarily a CPU correctness oracle.  It is much slower than
        m=2/m=3, but avoids a 4,963,410-coset flat table.
        """
        X, lead, matrix_input = self._as_matrix_batch(x)
        n = X.shape[0]
        if batch_chunk_size is None:
            batch_chunk_size = 1 if self.device.type == "cpu" else n
        if outer_a_chunk <= 0:
            raise ValueError("outer_a_chunk must be positive")

        out_m=[]; out_d=[]
        for blo in range(0,n,batch_chunk_size):
            Xb=X[blo:blo+batch_chunk_size]
            bsz=Xb.shape[0]
            best_score=torch.full((bsz,),float("-inf"),device=self.device,dtype=self.dtype)
            best_M=torch.zeros((bsz,8,8),device=self.device,dtype=self.dtype)
            for alo in range(0,270,outer_a_chunk):
                A=self.A2[alo:alo+outer_a_chunk]
                # Z[b,a] = A_a^T X_b.  For any M2 in S_2,
                # <X, A_a M2> = <A_a^T X, M2>.
                Z=torch.einsum("aji,bjk->baik",A,Xb)
                s=A.shape[0]
                Zf=Z.reshape(-1,8,8)
                # Keep the inner call memory bounded even if bsz*s is larger.
                inner_M=[]
                inner_d=[]
                inner_chunk=max(1,min(16,Zf.shape[0])) if self.device.type=="cpu" else Zf.shape[0]
                for zlo in range(0,Zf.shape[0],inner_chunk):
                    mm,dd=self._project_m2_branch_bound_flat(
                        Zf[zlo:zlo+inner_chunk],
                        initial_candidates=inner_initial_candidates,
                        pair_chunk_size=inner_pair_chunk_size,
                    )
                    inner_M.append(mm); inner_d.append(dd)
                M2=torch.cat(inner_M,0).reshape(bsz,s,8,8)
                # Direct correlation is cheap and avoids any dependence on
                # distance-roundoff when comparing the outer A choices.
                score=(Z*M2).sum(dim=(-2,-1))
                cs,ci=score.max(dim=1)
                improve=cs>best_score
                ar=torch.arange(bsz,device=self.device)
                cand=torch.matmul(A[ci],M2[ar,ci])
                best_M=torch.where(improve[:,None,None],cand,best_M)
                best_score=torch.where(improve,cs,best_score)
            d2=Xb.square().sum(dim=(-2,-1))+32.0-2.0*best_score
            out_m.append(best_M); out_d.append(torch.clamp_min(d2,0.0))

        M=torch.cat(out_m,0)
        d2=torch.cat(out_d,0)
        M=self._restore_matrix_shape(M,lead,matrix_input)
        if return_distance:
            return M,d2.reshape(lead)
        return M

    @torch.no_grad()
    def project_shell(
        self,
        x: torch.Tensor,
        m: int,
        *,
        batch_chunk_size: Optional[int] = None,
        transform_chunk: Optional[int] = None,
        return_distance: bool = False,
        algorithm: str = "auto",
    ):
        """Convenience dispatcher for the exact m=1,2,3,4 projectors."""
        if m == 1:
            return self.project_m1(
                x, batch_chunk_size=batch_chunk_size, transform_chunk=transform_chunk,
                return_distance=return_distance, algorithm=algorithm,
            )
        if m == 2:
            return self.project_m2(
                x, batch_chunk_size=batch_chunk_size, transform_chunk=transform_chunk,
                return_distance=return_distance, algorithm=algorithm,
            )
        if m == 3:
            return self.project_m3(
                x, batch_chunk_size=batch_chunk_size, transform_chunk=transform_chunk,
                return_distance=return_distance, algorithm=algorithm,
            )
        if m == 4:
            return self.project_m4(
                x, batch_chunk_size=batch_chunk_size, return_distance=return_distance,
            )
        raise ValueError("this package implements shell m=1, m=2, m=3, or m=4")

    # ------------------------- full lattice CVP -------------------------

    @torch.no_grad()
    def _costs_outer_chunk(self, X: torch.Tensor, outer_reps: torch.Tensor) -> torch.Tensor:
        """Old dense branch-cost builder, retained for validation/fallback."""
        bsz = X.shape[0]
        ucount = outer_reps.shape[0]
        T = X.transpose(-1, -2)[:, None, :, :] - 0.5 * outer_reps[None, :, None, :]
        costs = torch.empty((bsz, ucount, 8, 256), device=self.device, dtype=self.dtype)
        ar = self.e8_mod2_reps
        for j in range(8):
            tj = T[:, :, j, None, :]
            y = nearest_e8_mod_2(tj, ar[None, None, :, :])
            costs[:, :, j, :] = (tj - y).square().sum(dim=-1)
        return costs

    @torch.no_grad()
    def _decode_outer_chunk_dense(self, X: torch.Tensor, outer_ids: torch.Tensor):
        bsz = X.shape[0]
        outer_reps = self.e8_mod2_reps[outer_ids]
        ucount = outer_reps.shape[0]
        costs = self._costs_outer_chunk(X, outer_reps)
        dp = torch.full((bsz, ucount, 256), float("inf"), device=self.device, dtype=self.dtype)
        dp[:, :, 0] = 0.0
        choices = []
        for j in range(8):
            vals = dp[:, :, self.xor] + costs[:, :, j, None, :]
            dp, arg = vals.min(dim=-1)
            choices.append(arg.to(torch.uint8))
        terminal = dp[:, :, 0]
        local = terminal.argmin(dim=1)
        arange_b = torch.arange(bsz, device=self.device)
        best_cost = terminal[arange_b, local]
        best_outer = outer_ids[local]
        state = torch.zeros(bsz, dtype=torch.long, device=self.device)
        residues = torch.empty((bsz, 8), dtype=torch.long, device=self.device)
        for j in range(7, -1, -1):
            table = choices[j][arange_b, local].to(torch.long)
            r = table[arange_b, state]
            residues[:, j] = r
            state ^= r
        u = self.e8_mod2_reps[best_outer]
        T = X.transpose(-1, -2) - 0.5 * u[:, None, :]
        reps = self.e8_mod2_reps[residues]
        Yinner = nearest_e8_mod_2(T, reps)
        Y = (Yinner + 0.5 * u[:, None, :]).transpose(-1, -2).contiguous()
        return Y, best_cost, best_outer

    @torch.no_grad()
    def _cvp_dense_flat(self, X: torch.Tensor, outer_chunk_size: int):
        bsz = X.shape[0]
        best_cost = torch.full((bsz,), float("inf"), device=self.device, dtype=self.dtype)
        best_outer = torch.zeros((bsz,), dtype=torch.long, device=self.device)
        best_Y = torch.empty_like(X)
        ids = torch.arange(256, device=self.device)
        for lo in range(0, 256, outer_chunk_size):
            Y, cost, outer = self._decode_outer_chunk_dense(X, ids[lo:lo + outer_chunk_size])
            improve = cost < best_cost
            best_cost = torch.where(improve, cost, best_cost)
            best_outer = torch.where(improve, outer, best_outer)
            best_Y = torch.where(improve[:, None, None], Y, best_Y)
        return best_Y, best_cost, best_outer

    @torch.no_grad()
    def _inner_sparse_solve(
        self,
        T: torch.Tensor,
        branch_min: torch.Tensor,
        budget: torch.Tensor,
        *,
        pair_chunk_size: int = 512,
        backtrack: bool = False,
    ):
        """
        Exact syndrome-zero inner solve for selected outer cosets.

        Residues whose individual excess cost already exceeds the total feasible
        budget cannot occur in any improving solution, so the 256-way trellis is
        reduced to a small exact candidate list at each of the eight stages.
        """
        n = T.shape[0]
        out_cost, out_res = [], []
        states = torch.arange(256, device=self.device, dtype=torch.long)[None, :, None]
        for lo in range(0, n, pair_chunk_size):
            tt = T[lo:lo + pair_chunk_size]
            q = tt.shape[0]
            costs = self._e8_mod2_costs_fast(tt.reshape(-1, 8)).reshape(q, 8, 256)
            mins = branch_min[lo:lo + q]
            bud = torch.clamp_min(budget[lo:lo + q], 0.0)
            tol = self._tol(bud)[:, None, None]
            allowed = costs <= mins[:, :, None] + bud[:, None, None] + tol
            kmax = max(1, int(allowed.sum(dim=-1).max().item()))
            vals, rids = torch.topk(costs, k=kmax, dim=-1, largest=False, sorted=True)
            valid = vals <= mins[:, :, None] + bud[:, None, None] + tol

            dp = torch.full((q, 256), float("inf"), device=self.device, dtype=self.dtype)
            dp[:, 0] = 0.0
            choices = []
            for j in range(8):
                idx = torch.bitwise_xor(states, rids[:, j, None, :])
                prev = dp.gather(1, idx.reshape(q, -1)).reshape(q, 256, kmax)
                cand = prev + vals[:, j, None, :]
                cand = torch.where(valid[:, j, None, :], cand, torch.full_like(cand, float("inf")))
                if backtrack:
                    dp, aa = cand.min(dim=-1)
                    choices.append(aa.to(torch.uint8))
                else:
                    dp = cand.amin(dim=-1)
            out_cost.append(dp[:, 0])

            if backtrack:
                st = torch.zeros(q, device=self.device, dtype=torch.long)
                rr = torch.empty((q, 8), device=self.device, dtype=torch.long)
                ar = torch.arange(q, device=self.device)
                for j in range(7, -1, -1):
                    aa = choices[j][ar, st].to(torch.long)
                    rj = rids[ar, j, aa]
                    rr[:, j] = rj
                    st ^= rj
                out_res.append(rr)

        costs_out = torch.cat(out_cost, dim=0)
        if backtrack:
            return costs_out, torch.cat(out_res, dim=0)
        return costs_out

    @torch.no_grad()
    def _cvp_branch_bound_flat(
        self,
        X: torch.Tensor,
        *,
        seed_outer: int = 4,
        pair_chunk_size: int = 512,
    ):
        bsz = X.shape[0]
        reps = self.e8_mod2_reps
        T = X.transpose(-1, -2)[:, None, :, :] - 0.5 * reps[None, :, None, :]

        # Exact lower bound for every outer coset: ignore only the final residue
        # syndrome and decode each of the eight E8 columns independently.
        independent = e8_cvp(T)
        branch_min = (T - independent).square().sum(dim=-1)
        lower = branch_min.sum(dim=-1)

        # Residue of each independently decoded column.
        z = torch.round(independent @ self.Binv.T).to(torch.long) & 1
        rid = (z * self._residue_powers).sum(dim=-1)
        syndrome = rid[:, :, 0].clone()
        for j in range(1, 8):
            syndrome ^= rid[:, :, j]

        # Cheap feasible upper bound: repair the entire syndrome in one column.
        target_rid = rid ^ syndrome[:, :, None]
        target_rep = reps[target_rid]
        repaired = nearest_e8_mod_2(T, target_rep)
        repaired_cost = (T - repaired).square().sum(dim=-1)
        upper_outer = lower + (repaired_cost - branch_min).amin(dim=-1)
        best_cost, best_outer = upper_outer.min(dim=-1)
        ar = torch.arange(bsz, device=self.device)

        # Tighten the global feasible bound by exactly solving a few outer
        # cosets with the smallest lower bounds.  These results are reused.
        kseed = min(max(1, int(seed_outer)), 256)
        _, seed_ids = torch.topk(lower, kseed, dim=-1, largest=False, sorted=False)
        sb = ar[:, None].expand(bsz, kseed).reshape(-1)
        su = seed_ids.reshape(-1)
        seed_cost = self._inner_sparse_solve(
            T[sb, su], branch_min[sb, su], best_cost[sb] - lower[sb, su],
            pair_chunk_size=pair_chunk_size,
        ).reshape(bsz, kseed)
        sk = seed_cost.argmin(dim=-1)
        sval = seed_cost[ar, sk]
        souter = seed_ids[ar, sk]
        improve = sval < best_cost
        best_cost = torch.minimum(best_cost, sval)
        best_outer = torch.where(improve, souter, best_outer)

        seeded = torch.zeros((bsz, 256), device=self.device, dtype=torch.bool)
        seeded.scatter_(1, seed_ids, True)
        tol = self._tol(best_cost)[:, None]
        pairs = ((lower <= best_cost[:, None] + tol) & (~seeded)).nonzero(as_tuple=False)

        if pairs.numel():
            pb, pu = pairs[:, 0], pairs[:, 1]
            exact = self._inner_sparse_solve(
                T[pb, pu], branch_min[pb, pu], best_cost[pb] - lower[pb, pu],
                pair_chunk_size=pair_chunk_size,
            )
            # Survivor count is small; grouping by input is cheap and avoids a
            # dense [batch,256] exact-score table.
            for b in range(bsz):
                mm = pb == b
                if mm.any():
                    vv = exact[mm]
                    oo = pu[mm]
                    kk = vv.argmin()
                    if vv[kk] < best_cost[b]:
                        best_cost[b] = vv[kk]
                        best_outer[b] = oo[kk]

        # Reconstruct exact residues only for the winning outer coset.
        tb = T[ar, best_outer]
        bm = branch_min[ar, best_outer]
        lb = lower[ar, best_outer]
        check_cost, residues = self._inner_sparse_solve(
            tb, bm, best_cost - lb,
            pair_chunk_size=pair_chunk_size,
            backtrack=True,
        )
        # Use the reconstructed cost, which differs from best_cost only by
        # floating-point reduction order near ties.
        best_cost = check_cost
        u = reps[best_outer]
        rrep = reps[residues]
        Yinner = nearest_e8_mod_2(tb, rrep)
        Y = (Yinner + 0.5 * u[:, None, :]).transpose(-1, -2).contiguous()
        return Y, best_cost, best_outer

    @torch.no_grad()
    def cvp(
        self,
        x: torch.Tensor,
        *,
        batch_chunk_size: Optional[int] = None,
        outer_chunk_size: int = 4,
        return_info: bool = False,
        algorithm: str = "auto",
        seed_outer: int = 4,
        pair_chunk_size: int = 512,
    ):
        """Exact Euclidean CVP onto E8⊗E8."""
        X, lead, matrix_input = self._as_matrix_batch(x)
        n = X.shape[0]
        if algorithm == "auto":
            algorithm = "branch_bound" if self.device.type == "cpu" else "dense"
        if algorithm not in ("branch_bound", "dense"):
            raise ValueError("algorithm must be 'auto', 'branch_bound', or 'dense'")
        if batch_chunk_size is None:
            batch_chunk_size = 64 if (algorithm == "branch_bound" and self.device.type == "cpu") else n
        if batch_chunk_size <= 0:
            raise ValueError("batch_chunk_size must be positive")

        ys, costs, outers = [], [], []
        for lo in range(0, n, batch_chunk_size):
            Xi = X[lo:lo + batch_chunk_size]
            if algorithm == "branch_bound":
                Y, c, o = self._cvp_branch_bound_flat(
                    Xi, seed_outer=seed_outer, pair_chunk_size=pair_chunk_size,
                )
            else:
                Y, c, o = self._cvp_dense_flat(Xi, outer_chunk_size)
            ys.append(Y); costs.append(c); outers.append(o)

        Y = torch.cat(ys, dim=0)
        c = torch.cat(costs, dim=0)
        o = torch.cat(outers, dim=0)
        Y = self._restore_matrix_shape(Y, lead, matrix_input)
        if return_info:
            return Y, CVPInfo(c.reshape(lead), o.reshape(lead))
        return Y

    # ------------------------- diagnostics -------------------------

    @torch.no_grad()
    def is_lattice_point(self, y: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
        Y, lead, _ = self._as_matrix_batch(y)
        Z = self.Binv @ Y @ self.Binv.T
        err = (Z - torch.round(Z)).abs().amax(dim=(-2, -1))
        return (err <= atol).reshape(lead)

    @torch.no_grad()
    def shell_residual(self, M: torch.Tensor, m: int) -> torch.Tensor:
        X, lead, _ = self._as_matrix_batch(M)
        I = torch.eye(8, device=self.device, dtype=self.dtype)
        err = X.transpose(-1, -2) @ X - float(m) * I
        return err.abs().amax(dim=(-2, -1)).reshape(lead)


# -----------------------------------------------------------------------------
# Lightweight functional wrappers
# -----------------------------------------------------------------------------


@torch.no_grad()
def cvp_e8_tensor_e8(x: torch.Tensor, **kwargs):
    P = E8MatrixLattice(device=x.device, dtype=x.dtype if x.dtype.is_floating_point else torch.float32)
    return P.cvp(x, **kwargs)


@torch.no_grad()
def project_m1_shell(x: torch.Tensor, **kwargs):
    P = E8MatrixLattice(device=x.device, dtype=x.dtype if x.dtype.is_floating_point else torch.float32)
    return P.project_m1(x, **kwargs)


@torch.no_grad()
def project_m4_shell(x: torch.Tensor, **kwargs):
    P = E8MatrixLattice(device=x.device, dtype=x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32)
    return P.project_m4(x, **kwargs)


@torch.no_grad()
def project_m3_shell(x: torch.Tensor, **kwargs):
    P = E8MatrixLattice(device=x.device, dtype=x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32)
    return P.project_m3(x, **kwargs)


def project_m2_shell(x: torch.Tensor, **kwargs):
    P = E8MatrixLattice(device=x.device, dtype=x.dtype if x.dtype.is_floating_point else torch.float32)
    return P.project_m2(x, **kwargs)


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--skip-cvp", action="store_true")
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    P = E8MatrixLattice(args.device, dtype=dtype, preload_m2=True)
    X = torch.randn(args.batch, 8, 8, device=args.device, dtype=dtype)

    def timed(name, fn):
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize()
        print(f"{name:10s}: {time.perf_counter()-t0:.4f} s")
        return out

    M1, d1 = timed("m=1", lambda: P.project_m1(X, return_distance=True))
    M2, d2 = timed("m=2", lambda: P.project_m2(X, return_distance=True))
    print("m=1 residual:", float(P.shell_residual(M1, 1).max()))
    print("m=2 residual:", float(P.shell_residual(M2, 2).max()))
    print("m=1 lattice: ", bool(P.is_lattice_point(M1).all()))
    print("m=2 lattice: ", bool(P.is_lattice_point(M2).all()))

    if not args.skip_cvp:
        Y, info = timed("CVP", lambda: P.cvp(X, return_info=True))
        print("CVP lattice: ", bool(P.is_lattice_point(Y).all()))
        print("mean CVP d2: ", float(info.squared_error.mean()))
