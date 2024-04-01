module Kombinatorika where

import Test.QuickCheck
import Data.List

-- Marian Kravec

fact :: Int -> Int
fact n = product [1..n]

comb :: Int -> Int -> Int
comb n k = (fact n) `div` ((fact k) * (fact (n-k)))

-- permutacie
perms :: [t] -> [[t]]
perms [] = [[]]
perms (x:xs) = [insertInto x i ys | ys <- perms xs, i <- [0..length ys]]
                where
                insertInto x i xs = (take i xs) ++ (x:drop i xs)

qchPERM = quickCheck(\n -> (n > 0 && n < 10) ==> length (perms [1..n]) == fact n)

-- kombinacie bez opakovania
kbo :: [t] -> Int -> [[t]]
kbo _ 0 = [[]]
kbo (x:xs) k | (length xs)+1 < k = []
             | (length xs)+1 == k = [x:xs]
             | otherwise = [x:ys | ys <- (kbo xs (k-1))]++(kbo xs k)

qchKBO = quickCheck(\n1 -> \k1 -> let n = abs (n1 `mod` 10)
                                      k = abs (k1 `mod` n1)
                                  in n > 0 && k <= n ==> length (kbo [1..n] k) == comb n k)

-- kombinacie s opakovanim
kso :: [t] -> Int -> [[t]]
kso _ 0 = [[]]
kso (x:[]) k = [[x | i <- [1..k]]]
kso (x:xs) k = [x:ys | ys <- kso (x:xs) (k-1)]++(kso xs k)

qchKSO = quickCheck(\n1 -> \k1 -> let n = abs (n1 `mod` 10)
                                      k = abs (k1 `mod` n1)
                                  in n > 0 && k <= n ==> length (kso [1..n] k) == comb (n+k-1) k)

-- variacie bez opakovania
vbo :: (Eq t) => [t] -> Int -> [[t]]
vbo _ 0 = [[]]
vbo xs k | (length xs) < k = []
         | otherwise = [x:ys | x <- xs, ys <- (vbo xs (k-1)), not $ isIn x ys]
         where
           isIn x xs = (length $ filter (==x) xs) > 0

qchVBO = quickCheck(\n1 -> \k1 -> let n = abs (n1 `mod` 6)
                                      k = abs (k1 `mod` n1)
                                  in n > 0 && k <= n ==> length (vbo [1..n] k) == ((fact n) `div` (fact (n-k))))

-- variacie s opakovanim
vso :: [t] -> Int -> [[t]]
vso _ 0 = [[]]
vso xs k = [x:ys | x <- xs, ys <- (vso xs (k-1))]

qchVSO = quickCheck(\n1 -> \k1 -> let n = abs (n1 `mod` 8)
                                      k = abs (k1 `mod` n1)
                                  in n > 0 && k <= n ==> length (vso [1..n] k) == n^k)