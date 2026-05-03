//! PTX instruction text emission helpers.
//!
//! This module contains the [`Instruction::emit`] implementation and
//! the free helper functions for WMMA / MMA formatting. It lives as a
//! child module of `instruction` so that `super::*` resolves all of the
//! types defined there.

use super::{Instruction, MmaShape, Operand, Register, WmmaLayout, WmmaOp, WmmaShape};
use crate::ir::types::PtxType;

impl Instruction {
    /// Emits this instruction as a PTX assembly text string.
    ///
    /// The returned string includes the trailing semicolon (where applicable)
    /// and is suitable for direct inclusion in a PTX function body.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxicuda_ptx::ir::*;
    ///
    /// let dst = Register { name: "%f0".into(), ty: PtxType::F32 };
    /// let a = Operand::Register(Register { name: "%f1".into(), ty: PtxType::F32 });
    /// let b = Operand::Register(Register { name: "%f2".into(), ty: PtxType::F32 });
    /// let inst = Instruction::Add { ty: PtxType::F32, dst, a, b };
    /// assert_eq!(inst.emit(), "add.f32 %f0, %f1, %f2;");
    /// ```
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn emit(&self) -> String {
        match self {
            // -- Arithmetic -------------------------------------------------
            Self::Add { ty, dst, a, b } => {
                format!("add{} {dst}, {a}, {b};", ty.as_ptx_str())
            }
            Self::Sub { ty, dst, a, b } => {
                format!("sub{} {dst}, {a}, {b};", ty.as_ptx_str())
            }
            Self::Mul {
                ty,
                mode,
                dst,
                a,
                b,
            } => {
                format!(
                    "mul{}{} {dst}, {a}, {b};",
                    mode.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::Mad {
                ty,
                mode,
                dst,
                a,
                b,
                c,
            } => {
                format!(
                    "mad{}{} {dst}, {a}, {b}, {c};",
                    mode.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::MadLo { typ, dst, a, b, c } => {
                format!("mad.lo{} {dst}, {a}, {b}, {c};", typ.as_ptx_str())
            }
            Self::MadHi { typ, dst, a, b, c } => {
                format!("mad.hi{} {dst}, {a}, {b}, {c};", typ.as_ptx_str())
            }
            Self::MadWide {
                src_typ,
                dst,
                a,
                b,
                c,
            } => {
                format!("mad.wide{} {dst}, {a}, {b}, {c};", src_typ.as_ptx_str())
            }
            Self::Fma {
                rnd,
                ty,
                dst,
                a,
                b,
                c,
            } => {
                format!(
                    "fma{}{} {dst}, {a}, {b}, {c};",
                    rnd.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::Neg { ty, dst, src } => {
                format!("neg{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Abs { ty, dst, src } => {
                format!("abs{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Min { ty, dst, a, b } => {
                format!("min{} {dst}, {a}, {b};", ty.as_ptx_str())
            }
            Self::Max { ty, dst, a, b } => {
                format!("max{} {dst}, {a}, {b};", ty.as_ptx_str())
            }
            Self::Addc {
                ty,
                dst,
                a,
                b,
                carry_out,
            } => {
                let cc = if *carry_out { ".cc" } else { "" };
                format!("addc{cc}{} {dst}, {a}, {b};", ty.as_ptx_str())
            }
            Self::Selp {
                ty,
                dst,
                a,
                b,
                pred,
            } => {
                format!("selp{} {dst}, {a}, {b}, {pred};", ty.as_ptx_str())
            }

            // -- Bit Manipulation -------------------------------------------
            Self::Brev { ty, dst, src } => {
                format!("brev{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Clz { ty, dst, src } => {
                format!("clz{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Popc { ty, dst, src } => {
                format!("popc{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Bfind { ty, dst, src } => {
                format!("bfind{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Bfe {
                ty,
                dst,
                src,
                start,
                len,
            } => {
                format!("bfe{} {dst}, {src}, {start}, {len};", ty.as_ptx_str())
            }
            Self::Bfi {
                ty,
                dst,
                insert,
                base,
                start,
                len,
            } => {
                format!(
                    "bfi{} {dst}, {insert}, {base}, {start}, {len};",
                    ty.as_ptx_str()
                )
            }

            // -- Special Math -----------------------------------------------
            Self::Rcp { rnd, ty, dst, src } => {
                let rnd_str = rnd.map_or(String::new(), |r| r.as_ptx_str().to_string());
                format!("rcp{rnd_str}{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Rsqrt {
                approx,
                ty,
                dst,
                src,
            } => {
                let approx_str = if *approx { ".approx" } else { "" };
                format!("rsqrt{approx_str}{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Sqrt { rnd, ty, dst, src } => {
                let rnd_str = rnd.map_or(String::new(), |r| r.as_ptx_str().to_string());
                format!("sqrt{rnd_str}{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Ex2 {
                approx,
                ty,
                dst,
                src,
            } => {
                let approx_str = if *approx { ".approx" } else { "" };
                format!("ex2{approx_str}{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Lg2 {
                approx,
                ty,
                dst,
                src,
            } => {
                let approx_str = if *approx { ".approx" } else { "" };
                format!("lg2{approx_str}{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Sin {
                approx,
                ty,
                dst,
                src,
            } => {
                let approx_str = if *approx { ".approx" } else { "" };
                format!("sin{approx_str}{} {dst}, {src};", ty.as_ptx_str())
            }
            Self::Cos {
                approx,
                ty,
                dst,
                src,
            } => {
                let approx_str = if *approx { ".approx" } else { "" };
                format!("cos{approx_str}{} {dst}, {src};", ty.as_ptx_str())
            }

            // -- Shift operations -------------------------------------------
            Self::Shl {
                ty,
                dst,
                src,
                amount,
            } => {
                format!("shl{} {dst}, {src}, {amount};", ty.as_ptx_str())
            }
            Self::Shr {
                ty,
                dst,
                src,
                amount,
            } => {
                format!("shr{} {dst}, {src}, {amount};", ty.as_ptx_str())
            }

            // -- Integer Division & Modulo ----------------------------------
            Self::Div { ty, dst, a, b } => {
                format!("div{} {dst}, {a}, {b};", ty.as_ptx_str())
            }
            Self::Rem { ty, dst, a, b } => {
                format!("rem{} {dst}, {a}, {b};", ty.as_ptx_str())
            }

            // -- Bitwise Logic ----------------------------------------------
            Self::And { ty, dst, a, b } => {
                format!("and{} {dst}, {a}, {b};", ty.as_ptx_str())
            }
            Self::Or { ty, dst, a, b } => {
                format!("or{} {dst}, {a}, {b};", ty.as_ptx_str())
            }
            Self::Xor { ty, dst, a, b } => {
                format!("xor{} {dst}, {a}, {b};", ty.as_ptx_str())
            }

            // -- Comparison -------------------------------------------------
            Self::SetP { cmp, ty, dst, a, b } => {
                format!(
                    "setp{}{} {dst}, {a}, {b};",
                    cmp.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }

            // -- Memory -----------------------------------------------------
            Self::Load {
                space,
                qualifier,
                vec,
                ty,
                dst,
                addr,
            } => {
                format!(
                    "ld{}{}{}{} {dst}, {addr};",
                    space.as_ptx_str(),
                    qualifier.as_ptx_str(),
                    vec.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::Store {
                space,
                qualifier,
                vec,
                ty,
                addr,
                src,
            } => {
                format!(
                    "st{}{}{}{} {addr}, {src};",
                    space.as_ptx_str(),
                    qualifier.as_ptx_str(),
                    vec.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::CpAsync {
                bytes,
                dst_shared,
                src_global,
            } => {
                format!("cp.async.ca.shared.global [{dst_shared}], [{src_global}], {bytes};")
            }
            Self::CpAsyncCommit => "cp.async.commit_group;".to_string(),
            Self::CpAsyncWait { n } => {
                format!("cp.async.wait_group {n};")
            }

            // -- Type conversion --------------------------------------------
            Self::Cvt {
                rnd,
                dst_ty,
                src_ty,
                dst,
                src,
            } => {
                let rnd_str = rnd.map_or(String::new(), |r| r.as_ptx_str().to_string());
                format!(
                    "cvt{rnd_str}{}{} {dst}, {src};",
                    dst_ty.as_ptx_str(),
                    src_ty.as_ptx_str()
                )
            }

            // -- Control flow -----------------------------------------------
            Self::Branch { target, predicate } => match predicate {
                Some((pred, negated)) => {
                    let neg = if *negated { "!" } else { "" };
                    format!("@{neg}{pred} bra ${target};")
                }
                None => format!("bra ${target};"),
            },
            Self::Label(name) => format!("${name}:"),
            Self::Return => "ret;".to_string(),

            // -- Synchronization --------------------------------------------
            Self::BarSync { id } => format!("bar.sync {id};"),
            Self::BarArrive { id, count } => {
                format!("bar.arrive {id}, {count};")
            }
            Self::FenceAcqRel { scope } => {
                format!("fence.acq_rel{};", scope.as_ptx_str())
            }

            // -- Tensor Core: WMMA ------------------------------------------
            Self::Wmma {
                op,
                shape,
                layout,
                ty,
                fragments,
                addr,
                stride,
            } => emit_wmma(
                *op,
                *shape,
                *layout,
                *ty,
                fragments,
                addr.as_ref(),
                stride.as_ref(),
            ),

            // -- Tensor Core: MMA -------------------------------------------
            Self::Mma {
                shape,
                a_ty,
                b_ty,
                c_ty,
                d_ty,
                d_regs,
                a_regs,
                b_regs,
                c_regs,
            } => emit_mma(
                *shape, *a_ty, *b_ty, *c_ty, *d_ty, d_regs, a_regs, b_regs, c_regs,
            ),

            // -- Tensor Core: WGMMA -----------------------------------------
            Self::Wgmma {
                shape,
                d_ty,
                a_ty,
                b_ty,
                desc_a,
                desc_b,
                d_regs,
                scale_d,
                imm_scale_a,
                imm_scale_b,
                trans_a,
                trans_b,
            } => {
                let d_list = reg_list(d_regs);
                format!(
                    "wgmma.mma_async.sync.aligned{}{}{}{} {{{d_list}}}, {desc_a}, {desc_b}, {scale_d}, {imm_scale_a}, {imm_scale_b}, {trans_a}, {trans_b};",
                    shape.as_ptx_str(),
                    d_ty.as_ptx_str(),
                    a_ty.as_ptx_str(),
                    b_ty.as_ptx_str(),
                )
            }

            // -- TMA --------------------------------------------------------
            Self::TmaLoad {
                dst_shared,
                desc,
                coords,
                barrier,
            } => {
                let coord_list = coords
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [{dst_shared}], [{desc}, {{{coord_list}}}], [{barrier}];",
                )
            }

            // -- Atomic operations -------------------------------------------
            Self::Atom {
                space,
                op,
                ty,
                dst,
                addr,
                src,
            } => {
                format!(
                    "atom{}{}{} {dst}, [{addr}], {src};",
                    space.as_ptx_str(),
                    op.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::AtomCas {
                space,
                ty,
                dst,
                addr,
                compare,
                value,
            } => {
                format!(
                    "atom{}.cas{} {dst}, [{addr}], {compare}, {value};",
                    space.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::Red {
                space,
                op,
                ty,
                addr,
                src,
            } => {
                format!(
                    "red{}{}{} [{addr}], {src};",
                    space.as_ptx_str(),
                    op.as_ptx_str(),
                    ty.as_ptx_str()
                )
            }
            Self::AtomGlobalAddFloat { ty, dst, addr, src } => {
                format!("atom.global.add{} {dst}, [{addr}], {src};", ty.as_ptx_str())
            }

            // -- Special registers ------------------------------------------
            Self::MovSpecial { dst, special } => {
                format!("mov.u32 {dst}, {};", special.as_ptx_str())
            }

            // -- Parameter loading ------------------------------------------
            Self::LoadParam {
                ty,
                dst,
                param_name,
            } => {
                format!("ld.param{} {dst}, [{param_name}];", ty.as_ptx_str())
            }

            // -- Miscellaneous ----------------------------------------------
            Self::Comment(text) => format!("// {text}"),
            Self::Raw(text) => text.clone(),
            Self::Pragma(text) => format!(".pragma \"{text}\";"),

            // -- Video Instructions -----------------------------------------
            Self::Dp4a {
                dst,
                a,
                b,
                c,
                signed_a,
                signed_b,
            } => {
                let a_ty = if *signed_a { ".s32" } else { ".u32" };
                let b_ty = if *signed_b { ".s32" } else { ".u32" };
                format!("dp4a{a_ty}{b_ty} {dst}, {a}, {b}, {c};")
            }

            Self::Dp2a {
                dst,
                a,
                b,
                c,
                signed_a,
                signed_b,
                lo,
            } => {
                let a_ty = if *signed_a { ".s32" } else { ".u32" };
                let b_ty = if *signed_b { ".s32" } else { ".u32" };
                let half = if *lo { ".lo" } else { ".hi" };
                format!("dp2a{half}{a_ty}{b_ty} {dst}, {a}, {b}, {c};")
            }

            // -- PTX 8.x Instructions ---------------------------------------
            Self::Redux {
                op,
                dst,
                src,
                membership_mask,
            } => {
                format!(
                    "redux.sync{}.u32 {dst}, {src}, 0x{membership_mask:08x};",
                    op.as_ptx_str()
                )
            }
            Self::Stmatrix {
                dst_addr,
                src,
                shape,
                trans,
            } => {
                let trans_str = if *trans { ".trans" } else { "" };
                format!(
                    "stmatrix.sync.aligned{}{trans_str}.shared.b16 [{dst_addr}], {{{src}}};",
                    shape.as_ptx_str()
                )
            }
            Self::ElectSync {
                dst,
                membership_mask,
            } => {
                format!("elect.sync {dst}, 0x{membership_mask:08x};")
            }
            Self::Setmaxnreg { reg_count, action } => {
                format!("setmaxnreg{} {reg_count};", action.as_ptx_str())
            }
            Self::Griddepcontrol { action } => {
                format!("griddepcontrol{};", action.as_ptx_str())
            }
            Self::FenceProxy { scope, space } => {
                format!(
                    "fence.proxy.async{}{};",
                    scope.as_ptx_str(),
                    space.as_ptx_str()
                )
            }
            Self::MbarrierInit { addr, count } => {
                format!("mbarrier.init.shared.b64 [{addr}], {count};")
            }
            Self::MbarrierArrive { addr } => {
                format!("mbarrier.arrive.shared.b64 [{addr}];")
            }
            Self::MbarrierWait { addr, phase } => {
                format!("mbarrier.try_wait.parity.shared.b64 [{addr}], {phase};")
            }

            // -- SM 100+ (Blackwell) tcgen05 MMA ----------------------------
            Self::Tcgen05Mma { a_desc, b_desc } => {
                format!("tcgen05.mma.cta_group::1.kind::f32 [{a_desc}], [{b_desc}];")
            }

            // -- Cluster barrier / fence ------------------------------------
            Self::BarrierCluster => "barrier.cluster.arrive;".to_string(),
            Self::FenceCluster => "fence.mbarrier_init.release.cluster;".to_string(),

            // -- TMA bulk async copy -----------------------------------------
            Self::CpAsyncBulk {
                dst_smem,
                src_gmem,
                desc,
            } => {
                format!(
                    "cp.async.bulk.tensor.1d.shared::cluster.global.tile.bulk_group [{dst_smem}], [{src_gmem}, {{{desc}}}];"
                )
            }

            // -- Texture / Surface ------------------------------------------
            Self::Tex1d {
                ty,
                dst,
                tex_ref,
                coord,
            } => {
                format!(
                    "tex.1d.v4{}.s32 {dst}, [{tex_ref}, {{{coord}}}];",
                    ty.as_ptx_str()
                )
            }
            Self::Tex2d {
                ty,
                dst,
                tex_ref,
                coord_x,
                coord_y,
            } => {
                format!(
                    "tex.2d.v4{}.s32 {dst}, [{tex_ref}, {{{coord_x}, {coord_y}}}];",
                    ty.as_ptx_str()
                )
            }
            Self::Tex3d {
                ty,
                dst,
                tex_ref,
                coord_x,
                coord_y,
                coord_z,
            } => {
                format!(
                    "tex.3d.v4{}.s32 {dst}, [{tex_ref}, {{{coord_x}, {coord_y}, {coord_z}}}];",
                    ty.as_ptx_str()
                )
            }
            Self::SurfLoad {
                ty,
                dst,
                surf_ref,
                coord,
            } => {
                format!(
                    "suld.b.1d{} {dst}, [{surf_ref}, {{{coord}}}];",
                    ty.as_ptx_str()
                )
            }
            Self::SurfStore {
                ty,
                surf_ref,
                coord,
                src,
            } => {
                format!(
                    "sust.b.1d{} [{surf_ref}, {{{coord}}}], {src};",
                    ty.as_ptx_str()
                )
            }

            // -- ldmatrix (SM >= 75) ----------------------------------------
            Self::Ldmatrix {
                num_fragments,
                trans,
                dst_regs,
                src_addr,
            } => {
                let trans_str = if *trans { ".trans" } else { "" };
                let x_str = match num_fragments {
                    2 => ".x2",
                    4 => ".x4",
                    _ => ".x1",
                };
                let dst_list = dst_regs
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "ldmatrix.sync.aligned.m8n8{x_str}{trans_str}.shared.b16 {{{dst_list}}}, [{src_addr}];"
                )
            }
        }
    }
}

/// Emit a WMMA instruction family member.
#[allow(clippy::too_many_lines)]
fn emit_wmma(
    op: WmmaOp,
    shape: WmmaShape,
    layout: WmmaLayout,
    ty: PtxType,
    fragments: &[Register],
    addr: Option<&Operand>,
    stride: Option<&Operand>,
) -> String {
    let frag_list = reg_list(fragments);
    match op {
        WmmaOp::LoadA => {
            let addr_str = addr.map_or(String::new(), |a| format!("{a}"));
            let stride_str = stride.map_or(String::new(), |s| format!(", {s}"));
            format!(
                "wmma.load.a.sync.aligned{}{}{} {{{frag_list}}}, [{addr_str}]{stride_str};",
                shape.as_ptx_str(),
                layout.as_ptx_str(),
                ty.as_ptx_str()
            )
        }
        WmmaOp::LoadB => {
            let addr_str = addr.map_or(String::new(), |a| format!("{a}"));
            let stride_str = stride.map_or(String::new(), |s| format!(", {s}"));
            format!(
                "wmma.load.b.sync.aligned{}{}{} {{{frag_list}}}, [{addr_str}]{stride_str};",
                shape.as_ptx_str(),
                layout.as_ptx_str(),
                ty.as_ptx_str()
            )
        }
        WmmaOp::StoreD => {
            let addr_str = addr.map_or(String::new(), |a| format!("{a}"));
            let stride_str = stride.map_or(String::new(), |s| format!(", {s}"));
            format!(
                "wmma.store.d.sync.aligned{}{}{} [{addr_str}], {{{frag_list}}}{stride_str};",
                shape.as_ptx_str(),
                layout.as_ptx_str(),
                ty.as_ptx_str()
            )
        }
        WmmaOp::Mma => {
            format!(
                "wmma.mma.sync.aligned{}{}{} {{{frag_list}}};",
                shape.as_ptx_str(),
                layout.as_ptx_str(),
                ty.as_ptx_str()
            )
        }
    }
}

/// Emit an `mma.sync.aligned` instruction.
#[allow(clippy::too_many_arguments)]
fn emit_mma(
    shape: MmaShape,
    a_ty: PtxType,
    b_ty: PtxType,
    c_ty: PtxType,
    d_ty: PtxType,
    d_regs: &[Register],
    a_regs: &[Register],
    b_regs: &[Register],
    c_regs: &[Register],
) -> String {
    let d_list = reg_list(d_regs);
    let a_list = reg_list(a_regs);
    let b_list = reg_list(b_regs);
    let c_list = reg_list(c_regs);
    format!(
        "mma.sync.aligned{}.row.col{}{}{}{} {{{d_list}}}, {{{a_list}}}, {{{b_list}}}, {{{c_list}}};",
        shape.as_ptx_str(),
        d_ty.as_ptx_str(),
        a_ty.as_ptx_str(),
        b_ty.as_ptx_str(),
        c_ty.as_ptx_str()
    )
}

/// Format a comma-separated list of register names for use in `{...}` groups.
fn reg_list(regs: &[Register]) -> String {
    regs.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}
