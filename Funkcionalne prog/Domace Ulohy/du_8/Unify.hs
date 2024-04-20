module Unify where

import Terms
import Types
import TypeInference



contains :: Type -> Integer -> Bool
contains (Tvar i) j = (i == j)
contains (t1 :-> t2) j = (contains t1 j) || (contains t2 j)

substitute :: Integer -> Type -> Type -> Type
substitute i t (Tvar j) | i == j = t
                        | otherwise = Tvar j
substitute i t (t1 :-> t2) = ((substitute i t t1) :-> (substitute i t t2))

substitute' :: Integer -> Type -> Constraints -> Constraints
substitute' i t const = [(substitute i t c1, substitute i t c2) | (c1, c2) <- const]

add2Maybe :: Constraint -> Maybe Constraints -> Maybe Constraints
add2Maybe con (Just const) = (Just (con:const))
add2Maybe _ Nothing = Nothing

unify  :: Constraints -> Maybe Constraints
unify [] = Just []
unify (((Tvar i), (Tvar j)):xs) | i == j =  unify xs
                                | otherwise = add2Maybe ((Tvar i), (Tvar j)) (unify (substitute' i (Tvar j) xs))
unify (((Tvar i), (t1 :-> t2)):xs) | contains (t1 :-> t2) i = Nothing
                                   | otherwise = add2Maybe ((Tvar i), (t1 :-> t2)) (unify (substitute' i (t1 :-> t2) xs))
unify (((t1 :-> t2), (Tvar i)):xs) | contains (t1 :-> t2) i = Nothing
                                   | otherwise = add2Maybe ((Tvar i), (t1 :-> t2)) (unify (substitute' i (t1 :-> t2) xs))
unify (((t1 :-> t2), (t3 :-> t4)):xs) = unify ((t1, t3):(t2, t4):xs)


typeExp :: LExp -> Maybe Type
typeExp lterm = case unify (infType lterm) of
                     Just subst -> lookup (Tvar 0) subst
                     Nothing -> Nothing
