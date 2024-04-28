module HadajCis where

import Data.Char


hadam :: IO ()
hadam = do
          putStrLn ("zadaj tajne cislo:")
          strCis <- getLine
          putStrLn("pusti druheho hraca klast otazky")
          hadam' 0 ((read strCis)::Int)



hadam' :: Int -> Int -> IO ()
hadam' k cis = do
                  typ <- getLine
                  if cis == ((read typ)::Int)
                  then
                    putStrLn("uhadol na "++(show k)++" otazky")
                  else
                    if (cis `mod` ((read typ)::Int)) == 0
                    then
                      do
                       putStrLn("ano")
                       hadam' (k+1) cis
                    else
                      do
                       putStrLn("nie")
                       hadam' (k+1) cis

hadaj :: IO ()
hadaj = do
          putStrLn ("mysli si cislo 1..100")
          hadaj' 1 2 0

hadaj' :: Int -> Int -> Int -> IO()
hadaj' viem koef k = do
                    putStrLn ((show (viem*koef))++" ho deli")
                    odpoved <- getLine
                    if odpoved == "uhadol"
                    then putStrLn ("uhadol na "++(show k)++" otazky")
                    else
                      if odpoved == "ano"
                      then
                        do
                          hadaj' (viem*koef) 2 (k+1)
                      else
                        if odpoved == "nie"
                        then
                          do
                            hadaj' viem (koef+1) (k+1)
                        else
                          putStrLn ("nechapem odpovedi")
