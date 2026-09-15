param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [Parameter(Mandatory = $true)][string]$RecordPath,
    [int]$WorkerProcessId = 0,
    [long]$WorkerStartTicks = 0,
    [string]$QueueStatePath,
    [switch]$TestNotification,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$notificationConfig = Get-Content -Raw -Encoding UTF8 -LiteralPath $ConfigPath | ConvertFrom-Json
if ($notificationConfig.enabled -ne $true) { exit 0 }
$record = [ordered]@{
    status = 'initializing'
    created_at = [DateTime]::UtcNow.ToString('o')
    watcher_pid = $PID
    worker_pid = $WorkerProcessId
    expected_start_ticks = $WorkerStartTicks
    trigger = 'process_handle_wait'
    test_notification = [bool]$TestNotification
    dry_run = [bool]$DryRun
}

function Write-NotificationRecord {
    $directory = Split-Path -Parent $RecordPath
    [void][IO.Directory]::CreateDirectory($directory)
    $json = $record | ConvertTo-Json -Depth 6
    [IO.File]::WriteAllText($RecordPath, $json + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
}

function Send-QueueToast([string]$Kind) {
    $record['notification_kind'] = $Kind
    if ($DryRun) {
        $record['status'] = 'dry_run_completed'
        return
    }
    [void][Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime]
    [void][Windows.UI.Notifications.ToastNotifier, Windows.UI.Notifications, ContentType = WindowsRuntime]
    [void][Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime]
    [void][Windows.UI.Notifications.NotificationSetting, Windows.UI.Notifications, ContentType = WindowsRuntime]
    [void][Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime]
    $notifier = [Windows.UI.Notifications.ToastNotifier][Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($notificationConfig.app_id)
    if ($null -eq $notifier) { throw 'Windows did not create the notification sender.' }
    $record['windows_notification_setting'] = [string]$notifier.Setting
    $record['notifier_type'] = $notifier.GetType().FullName
    if ($record.windows_notification_setting -and $record.windows_notification_setting -ne 'Enabled') {
        throw ('Windows notifications are disabled: ' + $record.windows_notification_setting)
    }
    $message = $notificationConfig.messages.$Kind
    $title = [Security.SecurityElement]::Escape($message.title)
    $body = [Security.SecurityElement]::Escape($message.body)
    $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
    $xml.LoadXml("<toast><visual><binding template='ToastGeneric'><text>$title</text><text>$body</text></binding></visual></toast>")
    $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
    $toast.Tag = if ($TestNotification) { 'fais-test' } else { 'fais-' + $WorkerProcessId }
    $toast.Group = 'fais-queue'
    $toast.ExpirationTime = [DateTimeOffset]::Now.AddDays(1)
    $notifier.Show($toast)
    $record['toast_tag'] = $toast.Tag
    $record['status'] = 'submitted_to_windows'
}

try {
    if ($TestNotification) {
        Send-QueueToast 'test'
    } else {
        if ($WorkerProcessId -le 0 -or -not $QueueStatePath) {
            throw 'A specific worker and queue state are required.'
        }
        $workerProcess = $null
        try {
            $workerProcess = [Diagnostics.Process]::GetProcessById($WorkerProcessId)
        } catch {
            if ($_.Exception -isnot [ArgumentException] -and $_.Exception.InnerException -isnot [ArgumentException]) { throw }
        }
        $record['worker_exit_code'] = $null
        if ($null -ne $workerProcess) {
            # The Python timestamp can lose a few 100 ns units in conversion.
            if ($WorkerStartTicks -le 0 -or [Math]::Abs($workerProcess.StartTime.ToUniversalTime().Ticks - $WorkerStartTicks) -gt 10000) {
                throw 'The process start time differs; do not attach to a reused process ID.'
            }
            # Wait on the OS completion signal without a timer or progress reads.
            $null = $workerProcess.Handle
            $record['status'] = 'waiting_for_process_exit'
            Write-NotificationRecord
            $workerProcess.WaitForExit()
            $record['worker_exit_code'] = $workerProcess.ExitCode
        } else {
            $record['worker_already_exited'] = $true
        }
        $record['worker_exited_at'] = [DateTime]::UtcNow.ToString('o')
        $queue = Get-Content -Raw -Encoding UTF8 -LiteralPath $QueueStatePath | ConvertFrom-Json
        $record['queue_status'] = $queue.status
        $kind = 'interrupted'
        if ($queue.worker_pid -eq $WorkerProcessId) {
            if ($queue.status -eq 'completed' -and ($null -eq $record.worker_exit_code -or $record.worker_exit_code -eq 0) -and @($queue.pending_stages).Count -eq 0) {
                $kind = 'completed'
            } elseif ($queue.status -eq 'interrupted') {
                $kind = 'interrupted'
            } elseif ($queue.status -in @('failed', 'timed_out') -or ($null -ne $record.worker_exit_code -and $record.worker_exit_code -ne 0)) {
                $kind = 'failed'
            }
        }
        Send-QueueToast $kind
    }
    $record['finished_at'] = [DateTime]::UtcNow.ToString('o')
    Write-NotificationRecord
} catch {
    $record['status'] = 'notification_error'
    $record['error'] = $_.Exception.Message
    Write-NotificationRecord
    throw
}
