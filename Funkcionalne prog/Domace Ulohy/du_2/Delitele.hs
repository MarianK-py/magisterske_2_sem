module Delitele where
import Data.List

-- Marian Kravec

nesudelitelne  :: Int -> Int -> Bool
nesudelitelne x y | (x == 1) || (y == 1) = True
                  | ((x `mod` y) == 0) || ((y `mod` x) == 0) = False
                  | otherwise = disjunktne (delitele x) (delitele y)


disjunktne :: [Int] -> [Int] -> Bool
disjunktne xs ys = ((length xs)+(length ys)) == (length $ nub (xs++ys))

prvocislo :: Int -> Bool
prvocislo 1 = False
prvocislo n = (length $ delitele n) == 0

delitele :: Int -> [Int]
delitele x = sort $ nub $ delit 2
             where
               delit i | i <= (floor (sqrt (fromIntegral x))) = if ((x `mod` i) == 0)
                                                                then [i, (x `div` i)]++(delit (i+1))
                                                                else delit (i+1)
                       | otherwise = []