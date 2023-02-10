import SciLean.Tactic.CustomSimp.Main
-- import SciLean.Tactic.CustomSimp.AllPrePost
-- import SciLean.Tactic.CustomSimp.DebugSimp

import SciLean.Tactic.AutoDiff.LetDiff
-- import SciLean.AutoImpl
import SciLean.Core
-- import SciLean.Functions

-- import Lean.Meta
-- import Lean.Parser
-- import Lean.Elab

-- namespace Lean.Elab.Tactic
open Lean Meta Simp

namespace SciLean


partial def autoDiffPre (e : Expr) : SimpM Step := do
  -- trace[Meta.Tactic.simp] s!"Pre simp on:\n{← Meta.ppExpr e}"
  let e := e.headBeta

  -- Always use `autodiff` marked theorems and set them all as `pre` instead of post  
  let autodiffExt ← Lean.Meta.getSimpExtension? "autodiff"
  let mut autodiffThms ← autodiffExt.get!.getTheorems 
  autodiffThms := { autodiffThms with pre := autodiffThms.post, post := autodiffThms.pre }

  let allThms := #[autodiffThms].append (← read).simpTheorems
  for thms in allThms do
    if let some r ← rewrite? e thms.pre thms.erased DefaultMethods.discharge? (tag := "pre") (rflOnly := false) then
      -- trace[Meta.Tactic.simp] s!"Simplified to: {← Meta.ppExpr r.expr}"
      return ← andThen (Step.visit r) (λ e => autoDiffPre e)
  return Step.visit { expr := e }



-- Tactic

open Lean.Parser.Tactic in
syntax (name := autodiff_core) "autodiff_core " (config)? (discharger)? (&"only ")? ("[" (simpStar <|> simpErase <|> simpLemma),* "]")? (location)? : tactic

open Lean.Elab.Tactic in
@[tactic autodiff_core] def autoDiffCore : Tactic := fun stx => do
  let { ctx, dischargeWrapper } ← withMainContext <| mkSimpContext stx (eraseLocal := false)
  let usedSimps ← dischargeWrapper.with fun discharge? =>
    SciLean.Meta.CustomSimp.simpLocation ctx discharge? (expandOptLocation stx[5]) #[letDiff, autoDiffPre] #[]
  if tactic.simp.trace.get (← getOptions) then
    traceSimpCall stx usedSimps

-- Conv 

open Lean.Parser.Tactic in
syntax (name := autodiff_core_conv) "autodiff_core" (config)? (discharger)? (&" only")? (" [" (simpStar <|> simpErase <|> simpLemma),* "]")? : conv

open Lean.Elab.Tactic Lean.Elab.Tactic.Conv in
@[tactic autodiff_core_conv] def autoDiffCoreConv : Tactic := fun stx => withMainContext do
  let { ctx, dischargeWrapper, .. } ← mkSimpContext stx (eraseLocal := false)
  let lhs ← getLhs
  let (result, _) ← dischargeWrapper.with fun d? => SciLean.Meta.CustomSimp.simp lhs ctx (discharge? := d?) #[letDiff, autoDiffPre] #[]
  applySimpResult result


macro "autodiff" : conv => 
  `(conv| (autodiff_core (config := {singlePass := true,  zeta := false}) only []; 
           try simp (config := {zeta := false}) only [];))
macro "autodiff" : tactic => 
  `(tactic| (autodiff_core (config := {singlePass := true,  zeta := false}) only []; 
             try simp (config := {zeta := false}) only [];))

-- Tactic

open Lean.Parser.Tactic in
syntax (name := symdiff_core) "symdiff_core " (config)? (discharger)? (&"only ")? ("[" (simpStar <|> simpErase <|> simpLemma),* "]")? (location)? : tactic

open Lean.Elab.Tactic in
@[tactic symdiff_core] def symDiffCore : Tactic := fun stx => do
  let { ctx, dischargeWrapper } ← withMainContext <| mkSimpContext stx (eraseLocal := false)
  let usedSimps ← dischargeWrapper.with fun discharge? =>
    SciLean.Meta.CustomSimp.simpLocation ctx discharge? (expandOptLocation stx[5]) #[autoDiffPre] #[]
  if tactic.simp.trace.get (← getOptions) then
    traceSimpCall stx usedSimps

-- Conv 

open Lean.Parser.Tactic in
syntax (name := symdiff_core_conv) "symdiff_core" (config)? (discharger)? (&" only")? (" [" (simpStar <|> simpErase <|> simpLemma),* "]")? : conv

open Lean.Elab.Tactic Lean.Elab.Tactic.Conv in
@[tactic symdiff_core_conv] def symDiffCoreConv : Tactic := fun stx => withMainContext do
  let { ctx, dischargeWrapper, .. } ← mkSimpContext stx (eraseLocal := false)
  let lhs ← getLhs
  let (result, _) ← dischargeWrapper.with fun d? => SciLean.Meta.CustomSimp.simp lhs ctx (discharge? := d?) #[autoDiffPre] #[]
  applySimpResult result


macro "symdiff" : conv => 
  `(conv| (symdiff_core (config := {singlePass := true}) only [SciLean.tangentMap, SciLean.reverseDifferential]
           try simp (config := {zeta := false}) only [];))
macro "symdiff" : tactic => 
  `(tactic| (symdiff_core (config := {singlePass := true}) only [SciLean.tangentMap, SciLean.reverseDifferential]; 
             try simp (config := {zeta := false}) only [];))


set_option trace.Meta.Tactic.simp.rewrite true in
-- set_option trace.Meta.Tactic.simp.discharge true in
-- set_option trace.Meta.Tactic.simp.unify false in
#check (∂ λ (x : ℝ) => let y := x*x; y)
  rewrite_by
    autodiff
    trace_state


-- set_option trace.Meta.Tactic.simp true in
-- set_option trace.Meta.Tactic.simp.rewrite true in
-- set_option trace.Meta.Tactic.simp.discharge true in
-- set_option trace.Meta.Tactic.simp.unify false in
-- #check (∂ λ (x : ℝ) => let y := x*x; let z := x + y*x*x; x + y + z)
--   rewrite_by
--     autodiff_core (config:={zeta:=false}) only []
--     autodiff
--     trace_state

-- set_option trace.Meta.Tactic.simp true in
-- set_option trace.Meta.Tactic.simp.rewrite false in
-- set_option trace.Meta.Tactic.simp.unify false in
-- #check (∂ λ (x : ℝ) => let y := x; let z := x + y; y + z)
--   rewrite_by 
--     autodiff
--     trace_state

-- This fails to apply `SciLean.diff_of_comp` because it `foo` can't be proven to be smooth
-- set_option trace.Meta.Tactic.simp.rewrite true in
-- set_option trace.Meta.Tactic.simp.discharge true in
-- #check (∂ λ (x : ℝ) => let z := x^2; let foo := λ y => Math.sin (Math.exp y); foo (Math.cos z)) 
--   rewrite_by 
--     autodiff
--     trace_state


-- @[irreducible] def foo (a b : Nat) := a + b
-- @[simp] theorem foo_simp (a b : Nat) : foo a b = a + b := by unfold foo; rfl 

-- example {X Y Z W} [Vec X] [Vec Y] [Vec Z] [Vec W] (g : W → X) (h : W → Y) (f : W → X → Y → Z) [IsSmooth h] [IsSmooth g] [IsSmooth f] [∀ x, IsSmooth (f x)]  [∀ x y, IsSmooth (f x y)]
--   : (λ n : Nat => 
--     (∂ λ x w : W => 
--       let y := g x
--       let z := h x
--       f x y z))
--     =
--     λ n : Nat => 
--     hold
--     λ x dx w =>
--       let y  := g x
--       let dy := ∂ g x dx
--       let z  := h x
--       let dz := ∂ h x dx
--       ∂ f x dx y z + ∂ (f x) y dy z + ∂ (f x y) z dz
-- := by
--   autodiff_core (config := {zeta := false, singlePass := true})

--   -- autodiff_core (config := {zeta := false, singlePass := true})
--   -- autodiff_core (config := {zeta := false, singlePass := true})
--   -- autodiff_core (config := {zeta := false, singlePass := true})
--   simp[hold]
--   admit

-- @[simp]
-- theorem diff_at_zero {X Y} [Vec X] [Vec Y] (f : X → Y) [IsSmooth f] (x : X) : ∂ f x 0 = 0 := sorry

-- example {X Y} [Vec X] [Vec Y] (a : α) (g : α → X) (f : X → X → Y) [IsSmooth (λ x => f x (g a))]
--   : (λ n : Nat => (∂ λ x => 
--       let y := g a
--       f x y))
--     = 
--     λ n : Nat =>
--     hold
--     λ x dx =>
--       let y := g a
--       ∂ (λ x => f x y) x dx
-- := by
--   autodiff_core (config := {zeta := false})
--   simp[hold]
--   done

-- example {X Y} [Vec X] [Vec Y] (a : α) (g : α → X) (f : X → X → Y) [IsSmooth (f (g a))]
--   : (∂ λ x => 
--       let y := g a
--       f y x)
--     =
--     hold
--     λ x dx =>
--       let y  := g a
--       ∂ (f y) x dx
-- := by
--   autodiff_core (config := {zeta := false})
--   simp[hold]
--   done

