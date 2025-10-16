// Task 5.2.2: conversion of marks into grades, using a function
fun grade(mark: Int) = when (mark) {
    in 0..39   -> "Fail"
    in 40..69  -> "Pass"
    in 70..100 -> "Distinction"
    else       -> "?"
}

fun main(marks: Array<String>) {
    for (n in marks) {
        val integerGrade = n.toInt()
        val grade = grade(integerGrade)
        println("${integerGrade} is a ${grade}")
    }
}