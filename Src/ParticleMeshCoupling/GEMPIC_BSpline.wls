#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Section:: *)
(*GEMPIC  BSpline  Polynomials*)


Clear[GEMPICBSplineBasis]
(*
 The BSpline of Mathematica uses the definition
      | 1   0 < x < 1
 S^0 =|  
      | 0    else
 S^d = S^0*S^{d-1}
 which is a shift compared to GEMPIC definition:
 GEMPIC                          Mathematica
 [-(degree+1)/2,(degree+1)/2] -> [0,degree+1]
 The script utilizes the upper definition and the Mathematica range to shift all 
 BSpline polynomials into [0,1]
 
 Calculate the polynomials for the BSpline intervals which are implemented in GEMPIC
 All intervalls are mapped to [0,1].
 Usage: Set the BSpline degree below.
 Output: The script returns the BSpline as well as the first and second derivative
         with and without mapping the polynomials from an interval [n,n+1] -> [0,1]
*)
degree = 3;
GEMPICBSplineBasis[n_,x_] := PiecewiseExpand[BSplineBasis[n,1/(n+1)*x]];

Print["BSpline defined in the full support [0,degree+1]"]
GEMPICBSplineBasis[degree,x]
Plot[GEMPICBSplineBasis[degree,x],{x,-0.5,degree+1.5}]

Print["WARNING: The mapped polynomials have the same ordering of the intervals as the ones above which are NOT sorted!"]
Print["Polynomials of the BSpline shifted to the interval [0,1]"]
mappedBSplines=Map[Simplify@(#[[1]]/.x->(x+#[[2,1]]))&,
					GEMPICBSplineBasis[degree,x][[1]]]
Plot[mappedBSplines,{x,0,1}]

Print["First derivative of BSpline defined in the full support [0,degree+1]"]
bSplineFirstDerivative=D[GEMPICBSplineBasis[degree,x],x]
Plot[bSplineFirstDerivative,{x,-1,degree+2}]

Print["WARNING: The mapped polynomials have the same ordering of the intervals as the ones above which are NOT sorted!"]
Print["Polynomials of the first derivative of BSplines shifted to the interval [0,1]"]
mappedDerivativeBSplines=Map[Simplify@(#[[1]]/.x->(x+#[[2,1]]))&,
								D[GEMPICBSplineBasis[degree,x],x][[1]]]
Plot[mappedDerivativeBSplines,{x,0,1}]

Print["Second derivative of BSpline defined in the full support [0,degree+1]"]
bSplineSecondDerivative=D[GEMPICBSplineBasis[degree,x],{x,2}]
Plot[bSplineSecondDerivative,{x,-1,degree+2}]

Print["WARNING: The mapped polynomials have the same ordering of the intervals as the ones above which are NOT sorted!"]
Print["Polynomials of the second derivative of BSplines shifted to the interval [0,1]"]
mappedSecondDerivativeBSplines=Map[Simplify@(#[[1]]/.x->(x+#[[2,1]]))&,
									D[GEMPICBSplineBasis[degree,x],{x,2}][[1]]]
Plot[mappedSecondDerivativeBSplines,{x,0,1}]


Print["Primitives of BSplines"]
primitiveBSpline = Integrate[GEMPICBSplineBasis[degree,z],{z,0,x},Assumptions->x \[Element] Reals]
Plot[primitiveBSpline,{x,-0.5,degree+1.5}]

Print["WARNING: The mapped polynomials have the same ordering of the intervals as the ones above which are NOT sorted!"]
Print["Polynomials of the primitives of BSplines shifted to the interval [0,1]"]
mappedBSplinePrimitive = Map[Simplify@(#[[1]]/.x->(x+#[[2,1]]))&,Integrate[GEMPICBSplineBasis[degree,z],{z,0,x},Assumptions->x \[Element] Reals][[1]]]
Plot[mappedBSplinePrimitive,{x,0,1}]



