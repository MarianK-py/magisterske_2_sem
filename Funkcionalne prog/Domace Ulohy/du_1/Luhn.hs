module Luhn where

-- Marian Kravec

cardnumber :: Integer -> Bool
cardnumber n = ((accum n 0 1) `mod` 10) == 0
               where
                 accum n acc state | n == 0 = acc
                                   | otherwise = accum (n `div` 10) (acc + (if ((n `mod` 10) * state) < 10
                                                                            then ((n `mod` 10) * state)
                                                                            else ((n `mod` 10) * state)-9)) ((state `mod` 2)+1)