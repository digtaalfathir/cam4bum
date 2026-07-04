def determine_pattern(nums):
    def pattern(num):
        if num <= 44:
            return "FR" if num % 2 == 1 else "RR"
        elif (num - 45) % 3 == 0:
            return "FR"
        elif (num - 45) % 3 == 1:
            return "R1"
        else:
            return "R2"

    nums = [int(num) for num in nums]

    if len(nums) == 1:
        return pattern(nums[0])
    elif len(nums) == 2:
        pat = '-'.join([pattern(num) for num in nums])
    elif len(nums) == 3:
        pat = '-'.join([pattern(num) for num in nums[:2]]) + '-' + '-'.join([pattern(num) for num in nums[2:]])
    else:
        return "Invalid input, only accept 1, 2, or 3 numbers."

    return pat

# Hasil deteksi YOLO yang diubah menjadi list objek
detected_objects = ['25']
pattern_result = determine_pattern(detected_objects)

print(f"Pattern for detected objects {detected_objects} is '{pattern_result}'")


def test2():
    def pattern(num):
        if (num<=40) and (num>=1):
            return "FR" if num % 2 == 1 else "RR"
        elif num == 0:
            return "jig"
        elif (num - 41) % 3 == 0:
            return "FR"
        elif (num - 41) % 3 == 1:
            return "R1"
        else:
            return "R2"
    
    hasil = pattern(0)
    print (hasil)