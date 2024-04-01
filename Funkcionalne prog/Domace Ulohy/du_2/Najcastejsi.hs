module Najcastejsi where
import Data.List

-- Marian Kravec

najcastejsi :: (Ord t) => [t] -> t
najcastejsi xs = let
                   ys = nub xs
                   maxim x [] = x
                   maxim x (y:ys) = if (pocet x) > (pocet (maxim y ys)) then x else (maxim y ys)
                   pocet x = length $ filter (==x) xs
                 in
                   maxim (head ys) (tail ys)

najzriedkavejsi :: (Ord t) => [t] -> t
najzriedkavejsi xs = let
                       ys = nub xs
                       minim x [] = x
                       minim x (y:ys) = if (pocet x) < (pocet (minim y ys)) then x else (minim y ys)
                       pocet x = length $ filter (==x) xs
                     in
                       minim (head ys) (tail ys)

median :: (Integral t) => [t] -> t
median xs = (sort xs)!!(((length xs)-1) `div` 2)