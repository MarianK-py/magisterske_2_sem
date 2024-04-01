module Rozcvicka2 where
import Data.List
import Test.QuickCheck

-- Marian Kravec


-- pouzijem trochu iny pristup ako na prednaske a vzdy vytvorim iba slova
-- maju iny ako pociatocny znak
slovaSusedneRozne :: [Char] -> Int -> [String]
slovaSusedneRozne abc 0 = [[]]
slovaSusedneRozne abc 1 = [[x] | x <- abc]
slovaSusedneRozne abc k = [x:s | x <- abc, s <- slovaSusedneRozne abc (k-1), x /= head s]

-- najskor vygenerujeme vsetky slova, potom ich prefiltrujeme
-- tak, ze postupne znak po znaku z nasej abecedy vyhodime
-- slova ktore obsahuju dvojicu tychto znakov za sebou
slovaSusedneRozneFilter :: [Char] -> Int -> [String]
slovaSusedneRozneFilter abc k = overengineeredFilter abc (slova abc  k)
                                where
                                  overengineeredFilter [] ss = ss
                                  overengineeredFilter (x:xs) ss = overengineeredFilter xs (filter (not . isInfixOf [x,x]) ss)

-- z cvicenia, pouzite
slova :: String -> Int -> [String]
slova abeceda 0 = [ [] ]
slova abeceda n = [ ch:w | ch <- abeceda, w <- slova abeceda (n-1) ]

-- pocet slov vypocitame nasledovne:
-- na prvej pozicii moze byt hociake pismeno (length abeceda)
-- na kazdej dalsej moze byt hociake pismeno okrem toho ktore je prve v retazci (length abeceda -1)
-- celkovy pocet je sucin moznosti na jednotlivych poziciach cize
-- dlzka abececedy krát dlzky abecedy minus 1 (jedno pismeno ktore nesmieme pouzit) umocnene na (k-1)
-- (kedze riesimi vsetky pozicie okrem prvej)
pocetSlovaSusedneRozne :: [Char] -> Int -> Int
pocetSlovaSusedneRozne abc 0 = 1
pocetSlovaSusedneRozne abc k = let l = length abc in  l*((l-1)^(k-1))


-- quickCheck 1. vs. 3.
qchSlovaSusedneRozne = quickCheck(\abeceda -> \k -> (k >= 0 && k <= 10 &&
                                   1 < length abeceda && length abeceda < 6 &&
                                   length abeceda == length (nub abeceda)) ==> -- test, ktory diskvalifikuje abecedy s rovnakymi znakmi

                                   (length $ slovaSusedneRozne abeceda k) == (pocetSlovaSusedneRozne abeceda k)

                                   )


-- quickCheck 2. vs. 3.
qchSlovaSusedneRozneFilter = quickCheck(\abeceda -> \k -> (k >= 0 && k <= 10 &&
                                     1 < length abeceda && length abeceda < 6 &&
                                     length abeceda == length (nub abeceda)) ==> -- test, ktory diskvalifikuje abecedy s rovnakymi znakmi

                                     (length $ slovaSusedneRozneFilter abeceda k) == (pocetSlovaSusedneRozne abeceda k)

                                     )


-- u mna to preslo oboma testami

-- modifikovana verzia testov aby ich tolko nedropoval:
qch1vs3 = quickCheck(\abc1 -> \k1 -> let k = k1 `mod` 11
                                         abc = nub abc1
                                     in (length abc) < 6 ==> (length $ slovaSusedneRozne abc k) == (pocetSlovaSusedneRozne abc k))
qch2vs3 = quickCheck(\abc1 -> \k1 -> let k = k1 `mod` 11
                                         abc = nub abc1
                                     in (length abc) < 6 ==> (length $ slovaSusedneRozneFilter abc k) == (pocetSlovaSusedneRozne abc k))

