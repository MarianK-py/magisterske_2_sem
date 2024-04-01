module RozdielParnychNeparnych where

-- Marian Kravec

rozdielParnychNeparnych :: [Integer] -> Integer
rozdielParnychNeparnych zoz = abs (rozdiel zoz)
                              where
                                 rozdiel zoz = foldr (-) 0 zoz