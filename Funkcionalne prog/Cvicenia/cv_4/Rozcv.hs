module Rozcv where
  
import Data.List

-- Marian Kravec

nafukni :: [a] -> [a]
nafukni xs = foldr (\(x, i) acc -> (replicate i x)++acc) [] (zip xs [1..length xs])

nafukniR :: [a] -> [a]
nafukniR xs = fst $ foldl (\(acc, i) x -> (acc++(replicate i x), i-1)) ([], length xs) xs
