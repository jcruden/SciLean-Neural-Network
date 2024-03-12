import SciLean.Core.Notation.CDeriv
import SciLean.Core.FunctionTransformations.RevDeriv


--------------------------------------------------------------------------------
-- Notation  -------------------------------------------------------------------
--------------------------------------------------------------------------------

namespace SciLean.Notation

syntax "<∂ " term:66 : term
syntax "<∂ " diffBinder ", " term:66 : term
syntax "<∂ " "(" diffBinder ")" ", " term:66 : term

syntax "<∂! " term:66 : term
syntax "<∂! " diffBinder ", " term:66 : term
syntax "<∂! " "(" diffBinder ")" ", " term:66 : term

open Lean Elab Term Meta in
elab_rules : term
| `(<∂ $f $xs*) => do
  elabTerm (← `(revDeriv defaultScalar% $f $xs*)) none
| `(<∂ $f) => do
  elabTerm (← `(revDeriv defaultScalar% $f)) none
-- | `(<∂ $x:ident := $val:term ; $codir:term, $b) => do
--   let K := mkIdent (← currentFieldName.get)
--   elabTerm (← `(revDerivEval $K (fun $x => $b) $val $codir)) none

macro_rules
| `(<∂ $x:ident, $b) => `(<∂ (fun $x => $b))
| `(<∂ $x:ident := $val:term, $b) => `(<∂ (fun $x => $b) $val)
| `(<∂ $x:ident : $type:term, $b) => `(<∂ fun $x : $type => $b)
| `(<∂ ($b:diffBinder), $f)       => `(<∂ $b, $f)

macro_rules
| `(<∂! $f $xs*) => `((<∂ $f $xs*) rewrite_by autodiff)
| `(<∂! $f) => `((<∂ $f) rewrite_by autodiff)
| `(<∂! $x:ident, $b) => `(<∂! (fun $x => $b))
| `(<∂! $x:ident := $val:term, $b) => `(<∂! (fun $x => $b) $val)
| `(<∂! $x:ident : $type:term, $b) => `(<∂! fun $x : $type => $b)
| `(<∂! ($b:diffBinder), $f)       => `(<∂! $b, $f)


@[app_unexpander revDeriv] def unexpandRevDeriv : Lean.PrettyPrinter.Unexpander

  | `($(_) $_ $f:term $x) =>
    match f with
    | `(fun $x':ident => $b:term) => `(<∂ ($x':ident:=$x), $b)
    | `(fun ($x':ident : $_) => $b:term) => `(<∂ ($x':ident:=$x), $b)
    | _  => `(<∂ $f:term $x)

  | `($(_) $_ $f:term) =>
    match f with
    | `(fun $x':ident => $b:term) => `(<∂ $x':ident, $b)
    | `(fun ($x':ident : $ty) => $b:term) => `(<∂ ($x' : $ty), $b)
    | _  => `(<∂ $f)

  | _  => throw ()


-- @[app_unexpander revDerivEval] def unexpandRevDerivEval : Lean.PrettyPrinter.Unexpander

--   | `($(_) $_ $f:term $x $dy) =>
--     match f with
--     | `(fun $x':ident => $b:term) => `(<∂ ($x':ident:=$x;$dy), $b)
--     | `(fun ($x':ident : $_) => $b:term) => `(<∂ ($x':ident:=$x;$dy), $b)
--     | _  => throw ()

--   | _ => throw ()
