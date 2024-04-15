module SKI where

import Terms
import TermsSKI
--- some useful stuff
import Data.List 
import Data.Char
import Data.Map (Map, insert, lookup, empty, size)
import Data.Maybe

toSki :: LExp -> Ski
toSki exp = fromLExpSkitoSki (toSki'' (fromLExptoLExpSki exp))

free :: LExpSki -> [Var]
free (Id x) = [x]
free (App m n) = (free m)++(free n)
free (Lambda x b) = delete x (free b)
free (Ss) = []
free (Kk) = []
free (Ii) = []

toSki' :: LExpSki -> LExpSki
toSki' (Lambda x (Id y)) | x == y = (Ii)
                         | otherwise = (App (Kk) (Id y))
toSki' (Lambda x (App m n)) = (App (App (Ss) (toSki' (Lambda x m))) (toSki' (Lambda x n)))
toSki' (Lambda x y) = if not (elem x (free y))
                      then
                        (App (Kk) (toSki' y))
                      else
                        (Lambda x (toSki' y))
toSki' (App m n) = (App (toSki' m) (toSki' n))
toSki' (Id x) = (Id x)
toSki' (Ss) = (Ss)
toSki' (Kk) = (Kk)
toSki' (Ii) = (Ii)

toSki'' :: LExpSki -> LExpSki
toSki'' term = if isSki a then a else toSki'' a where a = toSki' term

isSki :: LExpSki -> Bool
isSki (Lambda x y) = False
isSki (Id x) = False
isSki (Ss) = True
isSki (Kk) = True
isSki (Ii) = True
isSki (App x y) = (isSki x) && (isSki y)

fromLExptoLExpSki :: LExp -> LExpSki
fromLExptoLExpSki (LAMBDA x b) = (Lambda x (fromLExptoLExpSki b))
fromLExptoLExpSki (ID x) = (Id x)
fromLExptoLExpSki (APP m n) = (App (fromLExptoLExpSki m) (fromLExptoLExpSki n))

fromLExpSkitoSki :: LExpSki -> Ski
fromLExpSkitoSki (Ss) = (S)
fromLExpSkitoSki (Kk) = (K)
fromLExpSkitoSki (Ii) = (I)
fromLExpSkitoSki (App m n) = (APL (fromLExpSkitoSki m) (fromLExpSkitoSki n))
fromLExpSkitoSki x = error ("syntax error")


fromSki :: Ski -> LExp
fromSki ski = fromLExpSkitoLExp (fromSki'' (fromSkitoLExpSki ski))

fromSki' :: LExpSki -> Int -> LExpSki
fromSki' (Ii) i =  (Lambda x (Id x))
                   where x = [['a'..'x']!!i]
fromSki' (App (Kk) c) i = (Lambda x (fromSki' c (i+1)))
                          where x = [[l | l<-['a'..'x'], not (elem [l] (free c))]!!i]


fromSki'' :: LExpSki -> LExpSki
fromSki'' term = if isLExp a then a else fromSki'' a where a = fromSki' term 0

isLExp :: LExpSki -> Bool
isLExp (Lambda x y) = True
isLExp (Id x) = True
isLExp (Ss) = False
isLExp (Kk) = False
isLExp (Ii) = False
isLExp (App x y) = (isLExp x) && (isLExp y)

fromSkitoLExpSki :: Ski -> LExpSki
fromSkitoLExpSki (S) = (Ss)
fromSkitoLExpSki (K) = (Kk)
fromSkitoLExpSki (I) = (Ii)
fromSkitoLExpSki (APL m n) = (App (fromSkitoLExpSki m) (fromSkitoLExpSki n))

fromLExpSkitoLExp :: LExpSki -> LExp
fromLExpSkitoLExp (Lambda x b) = (LAMBDA x (fromLExpSkitoLExp b))
fromLExpSkitoLExp (Id x) = (ID x)
fromLExpSkitoLExp (App m n) = (APP (fromLExpSkitoLExp m) (fromLExpSkitoLExp n))
fromLExpSkitoLExp x = error ("syntax error" )



oneStep ::  Ski -> Ski
oneStep (APL (APL (APL (S) f) g) x) = (APL (APL f x) (APL g x))
oneStep (APL (APL (K) c) x) = c
oneStep (APL I x) = x
oneStep (APL x y) = (APL (oneStep x) (oneStep y))
oneStep (S) = (S)
oneStep (K) = (K)
oneStep (I) = (I)



-- normalizator ako velkonocny darcek
nf :: Ski -> Ski
nf l = if a == l then a else nf a where a = oneStep l

a = LAMBDA "x" (LAMBDA "y" (APP (ID "y") (ID "x")))
b = LAMBDA "x" (LAMBDA "y" (APP (ID "x") (ID "y")))
c = LAMBDA "x" (LAMBDA "y" (LAMBDA "z" (APP (APP (ID "x") (ID "y")) (ID "z"))))

{- examples
toSki a
((S (K (S I))) ((S (K K)) I))
toSki b
((S ((S (K S)) ((S (K K)) I))) (K I))
toSki c
((S ((S (K S)) ((S (K (S (K S)))) ((S (K (S (K K)))) ((S ((S (K S)) ((S (K K)) I))) (K I)))))) (K (K I)))
toSki r

fromSki r
((\x->\y->\z->((x z) (y z)) \x->\y->x) (\x->\y->\z->((x z) (y z)) \x->\y->x))

fromSki s
((((((((\x->\y->x \x->\y->x) \x->\y->\z->((x z) (y z))) \x->\y->x) \x->\y->\z->((x z) (y z))) \x->\y->x) \x->\y->x) \x->\y->\z->((x z) (y z))) \x->\y->x)

nf du1
K
nf du2
S
-}
