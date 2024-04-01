module Podmnoziny where

-- Marian Kravec

podmnoziny :: [t] -> [[t]]
podmnoziny [] = [[]]
podmnoziny (x:xs) = let
                      ps = podmnoziny xs
                    in
                      ps++[x:p | p <- ps]

podmnozinyVPoradi :: [t] -> [[t]]
podmnozinyVPoradi [] = [[]]
podmnozinyVPoradi (x:xs) = let
                              ps = podmnozinyVPoradi xs
                           in
                              [x:p | p <- ps]++(reverse ps)