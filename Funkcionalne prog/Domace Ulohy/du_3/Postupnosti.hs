module Postupnosti where

-- Marian Kravec

nsrp   :: (Ord t) => [t] -> Int
nsrp  [] = -1
nsrp  (x:xs) = fst $ foldl (\(m, (p, x)) y -> if y > x then ((max m (p+1)), ((p+1), y)) else ((max m p), ((1, y)))) (1, (1, x)) xs


