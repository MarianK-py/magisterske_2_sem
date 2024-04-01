module Rok2020 where

-- Marian Kravec

dvojica :: [Integer] -> Integer
dvojica xs = dvojVseob xs 2020

stvorica:: [Integer] -> Integer
stvorica xs = stvorVseob xs 2020



dvojVseob:: [Integer] -> Integer -> Integer
dvojVseob xs n = maxim xs
             where
               maxim :: [Integer] -> Integer
               maxim [] = -1
               maxim (x:xs) = max (maxim' x xs) (maxim xs)

               maxim' :: Integer -> [Integer] -> Integer
               maxim' x [] = -1
               maxim' x (y:ys) | (x+y) == n = max (x*y) (maxim' x ys)
                               | otherwise = maxim' x ys

trojVseob:: [Integer] -> Integer -> Integer
trojVseob xs n = maxim xs
             where
               maxim :: [Integer] -> Integer
               maxim [] = -1
               maxim (x:xs) | ((n-x) >= 0) = max ((dvojVseob xs (n-x))*x) (maxim xs)
                            | otherwise = maxim xs

stvorVseob:: [Integer] -> Integer -> Integer
stvorVseob xs n = maxim xs
             where
               maxim :: [Integer] -> Integer
               maxim [] = -1
               maxim (x:xs) | ((n-x) >= 0) = max ((trojVseob xs (n-x))*x) (maxim xs)
                            | otherwise = maxim xs