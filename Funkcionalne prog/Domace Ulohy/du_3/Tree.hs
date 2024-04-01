module Tree where

data BVS t = Nil | Node (BVS t) t (BVS t) deriving(Show, Ord, Eq)

-- toto neefektivne riesenie bolo na prednaske
isBVS    :: (Ord t) => BVS t -> Bool
isBVS Nil = True
isBVS (Node left value right) =
      (all (<value) (flat left))
      &&
      (all (>value) (flat right))
      &&
      isBVS left
      &&
      isBVS right

findBVS  :: (Ord t) => t -> (BVS t) -> Bool
findBVS _ Nil = False
findBVS x (Node left value right)  | x == value = True
                                | x < value = findBVS x left
                                | x > value = findBVS x right

flat  :: BVS t -> [t]                                
flat Nil = []
flat (Node left value right)  = flat left ++ [value] ++ flat right
-------------------------------- tu konci prednaska, zvysok doprogramujte:

-- maximalny prvok neprazdneho binarneho vyhladavacieho stromu, ak splna podmienku ...
maxBVS   :: (Ord t) => BVS t -> t
maxBVS Nil = undefined
maxBVS (Node  l v Nil) = v
maxBVS (Node  l v r) = maxBVS r

-- minimalny prvok neprazdneho binarneho vyhladavacieho stromu, ak splna podmienku ...
minBVS    :: (Ord t) => BVS t -> t
minBVS Nil = undefined
minBVS (Node  Nil v r) = v
minBVS (Node  l v r) = minBVS l

-- linearna verzia isBVS, strom prejde len raz
isBVSLinear    :: (Ord t) => BVS t -> Bool
isBVSLinear Nil = True
isBVSLinear tr = isBVSLin' tr (minBVS tr) (maxBVS tr) []
                where
                  isBVSLin' Nil mini maxi inTree = True
                  isBVSLin' (Node l v r) mini maxi inTree | (v <= maxi) && (v >= mini) && (isBVSLin' l mini v (v:inTree)) && (isBVSLin' r v maxi (v:inTree)) && (not $ elem v inTree) = True
                                                          | otherwise = False

-- vlozenie, prvky v strome sa neopakuju
insertBVS      :: (Ord t) => t -> BVS t -> BVS t
insertBVS n Nil = Node Nil n Nil
insertBVS n (Node l v r) | n == v = Node l v r
                         | n < v = Node (insertBVS n l) v r
                         | otherwise = Node l v (insertBVS n r)

-- zmazanie
deleteBVS    :: (Ord t) => t -> BVS t -> BVS t
deleteBVS n Nil = Nil
deleteBVS n (Node Nil v Nil) | n == v = Nil
                             | otherwise = Node Nil v Nil
deleteBVS n (Node l v r) | n == v = let
                                      maxi = maxBVS l
                                      l1 = deleteBVS maxi l
                                    in
                                      Node l1 maxi r
                         | n < v = Node (deleteBVS n l) v r
                         | otherwise = Node l v (deleteBVS n r)
