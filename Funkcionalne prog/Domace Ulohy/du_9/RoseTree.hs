module RoseTree where
import Test.QuickCheck
import Control.Monad

data RoseTree a = Rose a [RoseTree a] deriving (Show, Eq)

instance Functor RoseTree where
    -- fmap :: (a -> b) -> List a -> List b
    fmap g (Rose a xs) = Rose (g a) (map (fmap g) xs)

instance Applicative RoseTree where
    -- pure :: a -> RoseTree a
   pure x = Rose x []
   -- (<*>) :: RoseTree (a -> b) -> RoseTree a -> RoseTree b
   (Rose a gs) <*> (Rose b xs) = Rose (a b) ([pure a <*> x | x <- xs]++[g <*> pure b | g <- gs]++[g <*> x | g <- gs, x <- xs])

-- sematanticke pravidla

instance Show (a -> b) where
         show a= "funcion"

oversized :: RoseTree a -> Int -> Bool
oversized (Rose a xs) maxS | (length xs) > maxS = True
                           | otherwise = any (\x -> oversized x maxS) xs

instance Arbitrary a => Arbitrary (RoseTree a) where
     arbitrary = frequency
                  [
                    (1, liftM2 Rose arbitrary arbitrary)
                  ]

qchIdentita = quickCheck((\tr -> (not $ oversized tr 2) ==> (pure id <*> tr) == tr)::RoseTree Int->Property)

qchKompozicia = quickCheck((\tr1 -> \tr2 -> \tr3 ->
                          ((not $ oversized tr1 3) && (not $ oversized tr2 3) && (not $ oversized tr3 3)) ==>
                          (pure (.) <*> tr1 <*> tr2 <*> tr3) == (tr1 <*> (tr2 <*> tr3)))::RoseTree (String->Int)->RoseTree (Int->String)->RoseTree Int->Property)

qchHomomorfizmus = quickCheck((\f -> \x -> (((pure f <*> pure x) :: RoseTree Int) == ((pure (f x)) :: RoseTree Int)))::(Int->Int)->Int->Bool)

qchVymena = quickCheck((\tr -> \y -> (not $ oversized tr 3) ==> ((tr <*> pure y) == (pure ($ y) <*> tr)))::RoseTree (Int->Int)->Int->Property)

-- musim sa naucit lepsie pracovat s interfacom Arbitrary