module CifSucet where

-- Marian Kravec

jCislaPocet :: Integer -> Integer -> Integer
jCislaPocet a b = ((b-a) `div` 9) + (accum a (a + ((b-a) `mod` 9)) 0)
                  where
                    accum i m acc | (i >= a) && (i <= m) = if (totalCifSuc i) == 5 then accum (i+1) m (acc+1) else accum (i+1) m acc
                                  | otherwise = acc

cifSuc :: Integer -> Integer
cifSuc n = accum n 0
           where
             accum n acc | n > 0 = accum (n `div` 10) (acc + (n `mod` 10))
                         | otherwise = acc

totalCifSuc :: Integer -> Integer
totalCifSuc n | (cifSuc n) >= 10 = totalCifSuc (cifSuc n)
              | otherwise = cifSuc n
