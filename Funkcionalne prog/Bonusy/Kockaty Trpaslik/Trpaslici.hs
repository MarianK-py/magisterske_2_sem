module Trpaslici where

-- Marian Kravec


pocetMoznosti :: Int -> Int
pocetMoznosti n = pocet 3 n 1
                  where
                    pocet 1 kocky minHrana = 1
                    pocet rozmer kocky minHrana = accum rozmer kocky minHrana 0

                    accum rozmer kocky i acc | i <= ceiling (sqrt (toEnum kocky)) = if (kocky `mod` i) == 0 && (kocky `div` i) >= i
                                                                                    then accum rozmer kocky (i+1) (acc+(pocet (rozmer-1) (kocky `div` i) i))
                                                                                    else accum rozmer kocky (i+1) acc
                                             | otherwise = acc

minim :: Int -> Integer
minim n = minim' 3 n 1 10000000000000000000 0 0
          where
            minim' :: Int -> Int -> Int -> Integer -> Int -> Int -> Integer
            minim' rozmer kocky i mini a b | rozmer == 1 = toInteger (2*((a*b)+(a*kocky)+(b*kocky)))
                                           | otherwise = if i <= ceiling (sqrt (toEnum kocky))
                                                         then
                                                           if (kocky `mod` i) == 0 && (kocky `div` i) >= i
                                                           then
                                                             if rozmer == 3
                                                             then
                                                               minim' rozmer kocky (i+1) (min mini (minim' (rozmer-1) (kocky `div` i) i mini i b)) a b
                                                             else
                                                               minim' rozmer kocky (i+1) (min mini (minim' (rozmer-1) (kocky `div` i) i mini a i)) a b
                                                           else
                                                             minim' rozmer kocky (i+1) mini a b
                                                         else
                                                           mini

maxim :: Int -> Integer
maxim n = maxim' 3 n 1 0 0 0
          where
            maxim' :: Int -> Int -> Int -> Integer -> Int -> Int -> Integer
            maxim' rozmer kocky i maxi a b | rozmer == 1 = toInteger (2*((a*b)+(a*kocky)+(b*kocky)))
                                           | otherwise = if i <= ceiling (sqrt (toEnum kocky))
                                                         then
                                                           if (kocky `mod` i) == 0 && (kocky `div` i) >= i
                                                           then
                                                             if rozmer == 3
                                                             then
                                                               maxim' rozmer kocky (i+1) (max maxi (maxim' (rozmer-1) (kocky `div` i) i maxi i b)) a b
                                                             else
                                                               maxim' rozmer kocky (i+1) (max maxi (maxim' (rozmer-1) (kocky `div` i) i maxi a i)) a b
                                                           else
                                                             maxim' rozmer kocky (i+1) maxi a b
                                                         else
                                                           maxi

maxim' :: Int -> Integer
maxim' n = (4*(toInteger n))+2

rozdiel :: Int -> Integer
rozdiel n = (maxim' n)-(minim n)