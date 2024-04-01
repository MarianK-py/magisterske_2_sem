module Bilandia2 where

-- Marian Kravec



dobryNakup :: Integer -> Bool--dobryNakup 0 = True
dobryNakup n = let
                 -- malo by stacit skontrolovat iba mocninu priamo nad
                 -- kdze dalsie su uz o 2^k vacsie a nemozu potom vratit
                 -- mocninu dvojky
                 log2 = ceiling $ logBase 2 (fromInteger n)
                 back = (2^log2) - n
               in
                 if back == 0
                 then True
                 else back == (2^(round $ logBase 2 (fromInteger back)))