module Main where

import Test.QuickCheck
import Text.Show.Functions
import Data.List(sort)


import Test.QuickCheck.Arbitrary
import Test.QuickCheck.Property (forAllShrink)
--import Data.Map hiding (map,null,filter,foldr)
--import Data.List (nub,delete)
--import Data.Data
--import Data.Char
import Control.Monad
--import Control.Monad.State
--import Data.Maybe (maybeToList)

import qualified Tree as F
import Test.HUnit
import System.Random
import System.IO.Unsafe

instance Arbitrary a => Arbitrary (F.BVS a) where
  arbitrary = frequency 
              [
                (1, return F.Nil )
              , (1, liftM3 F.Node arbitrary arbitrary arbitrary)
              ]
-- generate (arbitrary::Gen (BVS Int))

main = do
  g <- getStdGen
  runTestTT $  
      TestList [ 
        let tree = (F.Nil):: F.BVS Int in
            TestCase $ assertEqual ("basic test " ++ (show tree)) 
                                   True
                                   (F.isBVS tree),
        let treeList = map unsafePerformIO [generate (Test.QuickCheck.arbitrary::Gen (F.BVS Int)) | i<-[1..1000]] in
          TestList[
             TestCase $ assertEqual ("isBVSLinerar == isBVS" ++ (show tree)) 
                                   (F.isBVSLinear tree)
                                   (F.isBVS tree) | tree <-treeList],
        let treeListPair = [
                        (
                        unsafePerformIO $ generate (Test.QuickCheck.arbitrary::Gen (Int)) ,
                        unsafePerformIO $ generate (Test.QuickCheck.arbitrary::Gen (F.BVS Int)) 
                        ) | i<-[1..1000]] in
          TestList[
             TestCase $ assertEqual ("ak insertnem prvok a nasledne ho deletnem, dostanem povodny strom" ++ (show tree) ++ "," ++(show x)) 
                                   True
                                   (if F.isBVS tree && (F.findBVS x tree == False) then F.deleteBVS x (F.insertBVS x tree) == tree else True)
                                   | (x,tree) <-treeListPair],
        let treeListPair = [
                        (
                        unsafePerformIO $ generate (Test.QuickCheck.arbitrary::Gen (Int)) ,
                        unsafePerformIO $ generate (Test.QuickCheck.arbitrary::Gen (F.BVS Int)) 
                        ) | i<-[1..1000]] in
          TestList[
             TestCase $ assertEqual ("ak insertnem prvok, tak sa v nom nachadza" ++ (show tree)  ++ "," ++(show x)) 
                                   True
                                   (if F.isBVS tree then F.findBVS x (F.insertBVS x tree) else True)
                                   | (x,tree) <-treeListPair],
        let treeListPair = [
                        (
                        unsafePerformIO $ generate (Test.QuickCheck.arbitrary::Gen (Int)) ,
                        unsafePerformIO $ generate (Test.QuickCheck.arbitrary::Gen (F.BVS Int)) 
                        ) | i<-[1..1000]] in
          TestList[
             TestCase $ assertEqual ("ak deletnem prvok, tak sa v nom nenachadza" ++ (show tree)  ++ "," ++(show x)) 
                                   True
                                   (if F.isBVS tree then F.findBVS x (F.deleteBVS x tree) == False else True)
                                   | (x,tree) <-treeListPair],
        let treeListPair = [
                        (
                        unsafePerformIO $ generate (Test.QuickCheck.arbitrary::Gen (Int)) ,
                        unsafePerformIO $ generate (Test.QuickCheck.arbitrary::Gen (F.BVS Int)) 
                        ) | i<-[1..1000]] in
          TestList[
             TestCase $ assertEqual ("ak insertnem prvok do BVS, ostane to BVS" ++ (show tree)  ++ "," ++(show x)) 
                                   True
                                   (if F.isBVS tree then F.isBVS (F.insertBVS x tree) else True)
                                   | (x,tree) <-treeListPair],
        let treeListPair = [
                        (
                        unsafePerformIO $ generate (Test.QuickCheck.arbitrary::Gen (Int)) ,
                        unsafePerformIO $ generate (Test.QuickCheck.arbitrary::Gen (F.BVS Int)) 
                        ) | i<-[1..1000]] in
          TestList[
             TestCase $ assertEqual ("ak deletnem prvok do BVS, ostane to BVS" ++ (show tree)  ++ "," ++(show x)) 
                                   True
                                   (if F.isBVS tree then F.isBVS (F.deleteBVS x tree) else True)
                                   | (x,tree) <-treeListPair],
        let treeListPair = [
                        (
                        unsafePerformIO $ generate (Test.QuickCheck.arbitrary::Gen (Int)) ,
                        unsafePerformIO $ generate (Test.QuickCheck.arbitrary::Gen (F.BVS Int)) 
                        ) | i<-[1..1000]] in
          TestList[
             TestCase $ assertEqual ("x sa nachadza v strome, prave vtedy, ak sa nachadza vo flat stromu" ++ (show tree)  ++ "," ++(show x)) 
                                   True
                                   (if F.isBVS tree then F.findBVS x tree == elem x (F.flat tree) else True)
                                   | (x,tree) <-treeListPair],
        let lst = [unsafePerformIO $ generate (Test.QuickCheck.arbitrary::Gen (Int)) | i<-[1..1000]] in
          TestList[
             TestCase $ assertEqual ("ak insertnem 1..n do prazdneho stromu, stale dostanem BVS" ++ (show n)) 
                                   True
                                   (F.isBVS (foldr (\i -> \t -> F.insertBVS i t) (F.Nil) [1..n]))
                                   | n <-lst],
        let lst = [unsafePerformIO $ generate (Test.QuickCheck.arbitrary::Gen (Int)) | i<-[1..1000]] in
          TestList[
             TestCase $ assertEqual ("ak insertnem 1..n do prazdneho stromu, flat zo stromu bude mat dlzku n" ++ (show n)) 
                                   True
                                   (if n > 0 then length (F.flat (foldr (\i -> \t -> F.insertBVS i t) (F.Nil) [1..n])) == n else True)
                                   | n <-lst]
       ]
        
        
        
-- riesenie tutora

