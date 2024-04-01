module Skrinky where

-- Marian Kravec

pocetDel :: Integer -> Integer
pocetDel n | (floor (sqrt (fromInteger n)))^2 == n = 2*(1+(delit 2 0))-1
           | otherwise = 2*(1+(delit 2 0))
             where
               delit :: Integer -> Integer -> Integer
               delit i acc | i <= (floor (sqrt (fromInteger n))) = if (n `mod` i) == 0 then delit (i+1) (acc+1) else delit (i+1) acc
                           | otherwise = acc


pocetDel' :: Integer -> Integer
pocetDel' 1 = 1
pocetDel' n = delit n 2 0
              where
                delit :: Integer -> Integer -> Integer -> Integer
                delit n i acc | i < n = if (n `mod` i) == 0 then delit (n `div` i) i (acc+1) else delit n (i+1) acc
                              | i == n = acc+1
                              | otherwise = acc

neparnyPoc :: Integer -> Bool
neparnyPoc n = ((pocetDel n) `mod` 2) == 1

skrinky :: Integer -> Integer
skrinky n = foldr (+) 0 (filter (neparnyPoc) [1..n])