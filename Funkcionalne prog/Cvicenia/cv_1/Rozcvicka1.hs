module Rozcvicka1 where

-- Marian Kravec

-- podobne ako v desiatkovej sustave, vieme pocet cifier
-- pocitat ako dvojkovy logaritmus daneho cisla
-- jedine problemove pripady su ak n = 2^k kedze binlog(n)=k
-- ale 2^k ma az k+1 cifier, preto musime logaritmus cisla n+1
cifVBin :: Float -> Integer
cifVBin n = ceiling (logBase 2 (n+1))

-- pri pocitani cifier faktorialu vyuzijeme, ze log sucinu je sucet log
cifVBinFakt :: Float -> Integer
cifVBinFakt n = accum 1 0
                where
                  accum i acc | i <= n = accum (i+1) (acc + logBase 2 i)
                              | otherwise = ceiling acc
-- pre 1000! - 8530

-- pocet koncovych nul binarneho cisla zavisi od poctu dvojek v jeho
-- desiatkovom prvociselnom rozklade
koncNulyVBin :: Integer -> Integer
koncNulyVBin 0 = 1
koncNulyVBin n | (n `mod` 2) == 0 = 1+koncNulyVBin (n `div` 2)
               | otherwise = 0

-- podobne pre faktorial pouzijeme trik, ze budeme pocitat pocet dvojek
-- v rozklade kazdeho cisla v rozvoji faktorialu
koncNulyVBinFakt :: Integer -> Integer
koncNulyVBinFakt 0 = 1
koncNulyVBinFakt n = accum 1 0
                     where
                       accum i acc | i <= n = accum (i+1) (acc + koncNulyVBin i)
                                   | otherwise = acc
-- pre 1000! - 994

-- tu asi musíme prejst cele cislo a pocitat nuly
nulVBinZapise :: Integer -> Integer
nulVBinZapise n | n > 1 = if (n `mod` 2) == 0 then 1 + nulVBinZapise (n `div` 2) else nulVBinZapise (n `div` 2)
                | n == 1 = 0
                | otherwise = 1

-- pri pocitani tejto hodnoty pre faktorial... proste vypocitame faktorial
-- a z neho jeho pocet nul
nulVBinZapiseFakt :: Integer -> Integer
nulVBinZapiseFakt n = nulVBinZapise (fakt n)
                      where
                        fakt 0 = 1
                        fakt n = n*fakt (n-1)
-- pre 1000! - 4742