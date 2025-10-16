// Task 5.2.1: geometric properties of circles
import kotlin.math.PI

fun circleArea(radius: Double) = PI * radius * radius

fun circleCircumference(radius: Double) = PI * radius * 2.0

fun readDouble(prompt: String): Double {
    print(prompt)
    val number = readln().toDouble()
    return number
} 

fun main() {
    val radius = readDouble("Enter radius: ")
    println(circleArea(radius))
    println(circleCircumference(radius))
}