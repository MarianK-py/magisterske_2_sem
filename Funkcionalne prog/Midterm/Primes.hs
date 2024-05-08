module Primes where

-- Marian Kravec

-- O(n^(1/2))
prime :: Integer -> Bool
prime 1 = False
prime n = primeTest 2
          where
            primeTest :: Integer -> Bool
            primeTest i | i < (toInteger (fromEnum (sqrt (fromIntegral n))))+1 = if (n `mod` i) == 0 then False else primeTest (i+1)
                        | otherwise = True

primes :: Integer -> [Integer]
primes n = take (fromIntegral n) (filter (prime) [2..])

listMultPlusOne :: [Integer] -> Integer
listMultPlusOne l = (foldl (\acc x -> acc*x) 1 l)+1

prveZloz :: (Integer, Integer)
prveZloz = jeZloz 1
           where
             jeZloz :: Integer -> (Integer, Integer)
             jeZloz n = let
                          mult = listMultPlusOne (primes n)
                        in
                          if not (prime mult)
                          then (n, mult)
                          else jeZloz (n+1)

compMesner :: Integer -> Integer
compMesner n = (2^n)-1


prveNieMesner :: Integer
prveNieMesner = jeMesner 0
                where
                  primeNums = (filter (prime) [2..]) :: [Integer]
                  jeMesner :: Int -> Integer
                  jeMesner n | not (prime (compMesner (primeNums!!n))) = (primeNums!!n)
                             | otherwise = jeMesner (n+1)
