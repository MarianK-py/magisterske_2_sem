module Primes where

-- Marian Kravec

primes :: [Integer]
primes = map (isPrime) (nums)


nums :: [Integer]
nums = numgen 1
       where
         numgen :: Integer -> [Integer]
         numgen n = n:(numgen (n+1))

isPrime :: Integer -> Integer
isPrime 1 = 0
isPrime n = primeTest 2
            where
              primeTest :: Integer -> Integer
              primeTest i | i < n = if (n `mod` i) == 0 then 0 else primeTest (i+1)
                          | otherwise = n
