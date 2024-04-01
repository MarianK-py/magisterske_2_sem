module Rozcv where
  
import Data.List



-- Marian Kravec

-- pocet ciest A->Y prechadzajucich Q vieme zapisat
-- aj ako počet ciest A->Q krát pocet ciest Q->Y
-- (kedze cez Q musia vsetky prejst)
-- pozrime sa vseobacne na cesty x->y
-- uvazujme, ze musime prejst n krokov vpravo a m krokov dole
-- cize pocet ciest je pocet postupnosti tychot n a m krokov
-- tento pocet krokov vieme zapisat rekurzivnym vzorcom
-- K(n, m) = K(n-1, m) + K(n, m-1)
-- cize sucet poctu mosznosti ak pravime jeden krok vpravo
-- a ked spravime jeden krok nadol
-- pričom ak n == 0 alebo m == 0 tak existuje 1 cesta
-- ak si zacneme vyplnat tabulku dynamickeho programovanie vsimne si zaujimavu vec
-- .. 70 35 15 5 1
-- .. 35 20 10 4 1
-- .. 15 10  6 3 1
-- .. 5   4  3 2 1
-- .. 1   1  1 1 1
-- tieto hodnoty su napadne podobne hodnotam binomickeho trojuholnika
-- pricom ide o (n+m)-ty riadok a bud m-ta alebo n-ta pozicia co je kombinacne cislo (n+m) nad n

fact :: Integer -> Integer
fact n = product [1..n]

comb :: Integer -> Integer -> Integer
comb n k = (fact n) `div` ((fact k) * (fact (n-k)))

tab = [ "ABCDE","FGHIJ","KLMNO","PQRST","UVWXY"]

pocetSlov :: Char -> Char -> Char -> [String] -> Integer
pocetSlov a b c tab = let (ax, ay) = [(i, j) | i <- [0..length tab -1], j <- [0.. length (tab!!0) -1], tab!!i!!j == a]!!0
                          (bx, by) = [(i, j) | i <- [0..length tab -1], j <- [0.. length (tab!!0) -1], tab!!i!!j == b]!!0
                          (cx, cy) = [(i, j) | i <- [0..length tab -1], j <- [0.. length (tab!!0) -1], tab!!i!!j == c]!!0
                      in (pocty (toInteger (abs (ax-bx))) (toInteger (abs (ay-by)))) * (pocty (toInteger (abs (bx-cx))) (toInteger (abs (by-cy))))

-- ciest z A do Y cez Q je 16

pocetSlovCez :: (Integer, Integer) -> Integer -> Integer -> Integer
pocetSlovCez (cezX, cezY) n m = (pocty cezX cezY) * (pocty (n-cezX-1) (m-cezY-1))

-- kombinatoricky
pocty :: Integer -> Integer -> Integer
pocty n m = comb (m+n) n

-- rekurzivne
pocty' :: Integer -> Integer -> Integer
pocty' 0 m = 1
pocty' n 0 = 1
pocty' n m = (pocty (n-1) m) + (pocty n (m-1))

datumNar = [(i, j) | i <- [0..99], j <- [0..99], 57759367566058384683424532251391515779140098190920656000==(pocetSlovCez ((toInteger i), (toInteger j)) 100 100)]

-- typ: datum narodenia 28.12. co by malo byt znamenie: KOZOROZEC