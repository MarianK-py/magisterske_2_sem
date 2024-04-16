module DB where

import Terms
import TermsDB
--- some useful stuff
import Data.List 
import Data.Char
import Data.Map (Map, insert, lookup, empty, elems)
import Data.Maybe -- and maybe not

maxDepth :: LExp -> Int
maxDepth (ID _) = 0
maxDepth (APP e1 e2) = max (maxDepth e1) (maxDepth e2)
maxDepth (LAMBDA _ exp) = 1+(maxDepth exp)

maxDepthDB :: LExpDB -> Int
maxDepthDB (IDDB _) = 0
maxDepthDB (APPDB e1 e2) = max (maxDepthDB e1) (maxDepthDB e2)
maxDepthDB (LAMBDADB exp) = 1+(maxDepthDB exp)

type Indexes = Map String Int

maxIndex :: [Int] -> Int
maxIndex [x] = x
maxIndex (x:xs) =  max x (maxIndex xs)

toDB :: LExp -> LExpDB
toDB term = fst $ toDB' (0, maxDepth term) term empty

toDB' :: (Int, Int) -> LExp -> Indexes -> (LExpDB, Indexes)
toDB' (depth, maxdepth) (ID v) m = let vi = Data.Map.lookup v m
                                   in if isJust vi then -- viazana premenna
                                      (IDDB (depth - fromJust vi -1), m)
                                      else -- nenasiel, volna premenna, DU...
                                        let mi = maxIndex $ Data.Map.elems m
                                            vii = max (mi+1) maxdepth
                                        in (IDDB (vii), (Data.Map.insert v (vii) m))
toDB' (depth, maxdepth) (LAMBDA v lexp) m = let db = (toDB' ((depth+1), maxdepth) lexp (Data.Map.insert v depth m))
                                            in ((LAMBDADB (fst db)), (snd db))
toDB' (depth, maxdepth) (APP e1 e2) m =  let db1 = (toDB' (depth, maxdepth) e1 m)
                                             db2 = (toDB' (depth, maxdepth) e2 (snd db1))
                                         in (APPDB (fst db1) (fst db2), (snd db2))

fromDB :: LExpDB -> LExp
fromDB term = fromDB' term []

fromDB' :: LExpDB -> [String] -> LExp
fromDB' (APPDB e1 e2) m  = APP (fromDB' e1 m) (fromDB' e2 m)
fromDB' (IDDB index) m = if (index < length m) then
                               ID (m!!index)
                         else
                            ID [['a'..'z']!!index]
fromDB' (LAMBDADB exp) m = LAMBDA [['a'..'z']!!(length m)] (fromDB' exp ([['a'..'z']!!(length m)]:m))


subst :: LExpDB -> SubstDB -> LExpDB
subst (IDDB index) sub = sub !! index
subst (APPDB e1 e2) sub = APPDB (subst e1 sub) (subst e2 sub)
subst (LAMBDADB exp) sub = LAMBDADB $ subst exp sub' where sub' = (IDDB 0):map incr sub

incr :: LExpDB -> LExpDB
incr (IDDB i) = (IDDB (i+1))
incr lexp = incr' lexp (maxDepthDB lexp)

incr' :: LExpDB -> Int -> LExpDB
incr' (IDDB i) k | i >= k = (IDDB (i+1))
                 | otherwise = (IDDB i)
incr' (APPDB m n) i = (APPDB (incr' m i) (incr' n i))
incr' (LAMBDADB x) i = (LAMBDADB (incr' x i))


beta :: LExpDB -> LExpDB -> LExpDB
beta (LAMBDADB m) n = subst m (n:(map (\i -> IDDB i) [0..]))

-- velkonocny bonus
oneStep :: LExpDB -> LExpDB
oneStep (APPDB (LAMBDADB m) n) = beta (LAMBDADB (oneStep m)) (oneStep n)
oneStep (APPDB m n) = (APPDB (oneStep m) (oneStep n)) 
oneStep (LAMBDADB e) = LAMBDADB (oneStep e)
oneStep t@(IDDB i) = t


nf :: LExpDB -> LExpDB
nf t = if t == t' then t else nf t' where t' = oneStep t 

{-
toDB i = \0
toDB k = \\1
toDB s = \\\((2 0) (1 0))
-- foo = λz. ((λy. y (λx. x)) (λx. z x))
toDB foo = \(\(0 \0) \(1 0))
-- goo = (λx.λy.((z x) (λu.(u x)))) (λx.(w x))
toDB goo = (\\((3 1) \(0 2)) \(4 0)) ... voľná premenná
-- hoo = λx.λy.y (λz.z x) x
toDB hoo = \\((0 \(0 2)) 1)
-- ioo = λx.(λx.x x) (λy.y (λz.x))
toDB ioo = \(\(0 0) \(0 \2))

fromDB $ toDB i = \x->x
fromDB $ toDB k = \x->\y->x
fromDB $ toDB s = \x->\y->\z->((x z) (y z))
fromDB $ toDB foo = \x->(\y->(y \z->z) \y->(x y))
fromDB $ toDB goo = (\x->\y->((d x) \z->(z x)) \x->(e x)) ... voľná premenná
fromDB $ toDB hoo = \x->\y->((y \z->(z x)) x)
fromDB $ toDB ioo = \x->(\y->(y y) \y->(y \z->x))
-}


{- examples toDB
toDB i
\0
toDB k
\\1
toDB s
\\\((2 0) (1 0))
toDB foo
\(\(0 \0) \(1 0))
toDB goo
(\\((3 1) \(0 2)) \(4 0))
toDB hoo
\\((0 \(0 2)) 1)
toDB ioo
\(\(0 0) \(0 \2))
toDB izero
\\0
toDB omega
\(0 0)
toDB isucc
\\\(1 ((2 1) 0))
toDB y
\(\(1 (0 0)) \(1 (0 0)))
toDB omega3
\((0 0) 0)
toDB bigOmega
(\(0 0)
toDB ifour
(\\\(1 ((2 1) 0)) (\\\(1 ((2 1) 0)) (\\\(1 ((2 1) 0)) (\\\(1 ((2 1) 0)) \\0))))
toDB iplus
\\\\((3 1) ((2 1) 0))
toDB itimes
\\\\((3 (2 1)) 0)
toDB ipower
\\(0 1)
-}



{- examples fromDB
fromDB $ toDB i
\x->x
fromDB $ toDB k
\x->\y->x
fromDB $ toDB s
\x->\y->\z->((x z) (y z))
fromDB $ toDB foo
\x->(\y->(y \z->z) \y->(x y))
fromDB $ toDB goo
(\x->\y->((d x) \z->(z x)) \x->(e x))
fromDB $ toDB hoo
\x->\y->((y \z->(z x)) x)
fromDB $ toDB ioo
\x->(\y->(y y) \y->(y \z->x))
fromDB $ toDB izero
\x->\y->y
fromDB $ toDB omega
\x->(x x)
fromDB $ toDB isucc
\x->\y->\z->(y ((x y) z))
fromDB $ toDB y
\x->(\y->(x (y y)) \y->(x (y y)))
fromDB $ toDB omega3
\x->((x x) x)
fromDB $ toDB bigOmega
(\x->(x x) \x->(x x))
fromDB $ toDB ifour
(\x->\y->\z->(y ((x y) z)) (\x->\y->\z->(y ((x y) z)) (\x->\y->\z->(y ((x y) z)) (\x->\y->\z->(y ((x y) z)) \x->\y->y))))
fromDB $ toDB iplus
\x->\y->\z->\w->((x z) ((y z) {))
fromDB $ toDB itimes
\x->\y->\z->\w->((x (y z)) {)
fromDB $ toDB ipower
\x->\y->(y x)

-}


{- examples nf
 nf $ toDB ione
\\(1 0)
 nf $ toDB itwo
\\(1 (1 0))
 nf $ toDB ifour
\\(1 (1 (1 (1 0))))
 nf $ toDB iquad
\\(1 (1 (1 (1 0))))
 nf $ toDB ieight
\\(1 (1 (1 (1 (1 (1 (1 (1 0))))))))
 nf $ toDB isix
\\(1 (1 (1 (1 (1 (1 0))))))
 nf $ toDB ithree
\\(1 (1 (1 0)))
 nf $ toDB inine
\\(1 (1 (1 (1 (1 (1 (1 (1 (1 0)))))))))
 nf $ toDB isixteen
\\(1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 0))))))))))))))))
-}
