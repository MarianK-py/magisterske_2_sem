module KSucin where

cart  :: [[t]] -> [[t]]
cart xs = foldr (\x acc -> pure (:) <*> x <*> acc) [[]] xs