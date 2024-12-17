def voting_simulation(cumulative_predictions : list ) -> list:  
    """This methods simulates the voting among predictions of all the models and returns the final output"""
    final_output = []

    for i in range(0, len(cumulative_predictions[0])):
        count_one = 0
        count_zero = 0

        d = []
        for x in range(0, len(cumulative_predictions)):
            d.append(cumulative_predictions[x][i])

        for j in range(0,len(d)):
            if d[j] == 1:
                count_one = count_one+1
            else:
                count_zero = count_zero+1
        if count_one > count_zero:
            final_output.append(1)
        else:
            final_output.append(0)
    return final_output

def main():
    """main function"""

if __name__ == "__main__":
    main()