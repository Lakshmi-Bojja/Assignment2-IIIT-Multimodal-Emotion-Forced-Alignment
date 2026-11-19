Add-Type -AssemblyName System.Speech
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speak.Rate = -1
$speak.Volume = 100
" 【entity-Hennessy¦canonical_name=Hennessy】 was the chief justice of the Massachusetts Supreme Court." | Out-File -Encoding ascii corpus\spk1\hennessy_1.lab
" 【entity-Hennessy¦canonical_name=Hennessy】 was widely respected for his legal scholarship." | Out-File -Encoding ascii corpus\spk1\hennessy_2.lab
" 【entity-Hennessy¦canonical_name=Hennessy】 led the court for many years with great care." | Out-File -Encoding ascii corpus\spk1\hennessy_3.lab
"I SAID BAD NOT BED" | Out-File -Encoding ascii corpus\spk1\minimal_bad_bed.lab
"I SAID NEW NOT NO" | Out-File -Encoding ascii corpus\spk1\minimal_new_no.lab
"I SAID WHITE NOT BAIT" | Out-File -Encoding ascii corpus\spk1\minimal_white_bait.lab
$speak.SetOutputToWaveFile("corpus\spk1\hennessy_1.wav")
$speak.Speak(" 【entity-Hennessy¦canonical_name=Hennessy】 was the chief justice of the Massachusetts Supreme Court.")
$speak.SetOutputToWaveFile("corpus\spk1\hennessy_2.wav")
$speak.Speak(" 【entity-Hennessy¦canonical_name=Hennessy】 was widely respected for his legal scholarship.")
$speak.SetOutputToWaveFile("corpus\spk1\hennessy_3.wav")
$speak.Speak(" 【entity-Hennessy¦canonical_name=Hennessy】 led the court for many years with great care.")
$speak.SetOutputToWaveFile("corpus\spk1\minimal_bad_bed.wav")
$speak.Speak("I said bad not bed.")
$speak.SetOutputToWaveFile("corpus\spk1\minimal_new_no.wav")
$speak.Speak("I said new not no.")
$speak.SetOutputToWaveFile("corpus\spk1\minimal_white_bait.wav")
$speak.Speak("I said white not bait.")
$speak.SetOutputToNull()
Write-Host "DONE! 6 .wav files created!"