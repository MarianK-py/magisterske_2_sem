module TypeInference where

import Terms
import Types

infType t = inferType [(t, Tvar 0)] [] [] 1

lookupType :: Context -> String -> Type
lookupType [] s = Tvar (-1)
lookupType ((s1, t):xs) s2 | s1 == s2 = t
                           | otherwise = lookupType xs s2

inferType :: [(LExp,Type)]->Context->Constraints->Integer->Constraints
inferType [] ctx const i = const
inferType (((ID x), t):xs) ctx const i = let
                                          t1 = lookupType ctx x
                                         in
                                          inferType xs ctx ((t, t1):const) i
inferType (((LAMBDA x lexp), t):xs) ctx const i = let
                                                    t1 = Tvar i
                                                    t2 = Tvar (i+1)
                                                  in
                                                    inferType ((lexp, t2):xs) ((x, t1):ctx) ((t, (t1 :-> t2)):const) (i+2)
inferType (((APP m n), t):xs) ctx const i = let
                                              t1 = Tvar i
                                              t2 = Tvar (i+1)
                                            in
                                              inferType ((m, t1):(n, t2):xs) ctx ((t1, (t2 :-> t)):const) (i+2)