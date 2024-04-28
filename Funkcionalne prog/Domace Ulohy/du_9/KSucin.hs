module KSucin where

cart  :: [[t]] -> [[t]]
cart xs = foldr (\x acc -> pure (:) <*> x <*> acc) [[]] xs

-- uprimne mi nenapada ako napisat verziu ktora by pouzivala Cantorovu parovaciu funkciu