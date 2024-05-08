module Funkc where

import Data.List
-- Marian Kravec


scalar :: [Int] -> [Int] -> Int
scalar a b = foldl (\acc (i, j) -> acc+(i*j)) 0 (zip a b)

add :: [[Int]] -> [[Int]] -> [[Int]]
add a b = foldl (\acc (i, j) -> acc++[(zipWith (+) i j)]) [] (zip a b)

cart2 :: [t] -> [s] ->[(t,s)]
cart2 a b = pure (,) <*> a <*> b


mult :: [[Int]] -> [[Int]] -> [[Int]]
mult a b = let
             bt = transpose b
             crt = cart2 a bt
             mat = map (\(i,j) -> scalar i j) crt
             n = length a
             (vys_mat,_,_) = foldl (\(acc, temp_acc, k) x -> if k < n-1
                                                             then (acc, temp_acc++[x], k+1)
                                                             else (acc++[temp_acc++[x]], [], 0)) ([], [], 0) mat
           in
             vys_mat