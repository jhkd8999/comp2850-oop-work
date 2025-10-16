// Task 5.1.2: die rolling simulation
import kotlin.random.Random

fun rollDie(sides: Int = 6) {
    if (sides in setOf(4, 6, 8, 10, 12, 20)) {
        println("Rolling a d$sides...")
        val result = Random.nextInt(1, sides + 1)
        println("You rolled $result")
    }
    else {
        println("Error: cannot have a $sides-sided die")
    }
}

fun readInt(string: String): Int {
    println("You chose ${string}")
    return string.toInt()
}

fun main() {
    println("Function with no input (should roll a d6)")
    rollDie()
    println("How many sides for die: ")
    var side = readln()
    rollDie(readInt(side))
}