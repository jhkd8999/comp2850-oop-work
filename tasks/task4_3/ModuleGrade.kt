import kotlin.math.roundToInt
import kotlin.system.exitProcess

fun main(args: Array<String>) {
    if (args.size != 3) {
        println("Error: 3 arguments required")
        exitProcess(1)
    }
    val average = ((args[0].toInt() + args[1].toInt() + args[2].toInt())/3.0).roundToInt()

    val grade = when (average) {
        in 0..39 -> "Fail"
        in 40..69 -> "Pass"
        in 70..100 -> "Distinction"
        else -> "?"
    }
    println("Average mark of ${average} is grade ${grade}")
}   