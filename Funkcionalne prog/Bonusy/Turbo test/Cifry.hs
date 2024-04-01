module Cifry where

-- Marian Kravec

cifry :: Integer -> [Integer]
cifry n = accum n []
          where
            accum :: Integer -> [Integer] -> [Integer]
            accum n cif | n < 10 = [n] ++ cif
                        | otherwise = accum (n `div` 10) ([(n `mod` 10)] ++ cif)

cifryR :: Integer -> [Integer]
cifryR n = reverse (cifry n)