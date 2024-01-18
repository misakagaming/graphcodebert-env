// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: MYY10151_OA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:42:02
//
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// using Statements
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
using System;
using com.ca.gen.vwrt;
using com.ca.gen.vwrt.types;
using com.ca.gen.vwrt.vdf;
using com.ca.gen.csu.exception;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// START OF EXPORT VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: MYY10151_OA
  /// </summary>
  [Serializable]
  public class MYY10151_OA : ViewBase, IExportView
  {
    private static MYY10151_OA[] freeArray = new MYY10151_OA[30];
    private static int countFree = 0;
    
    // Repeating GV:  EXP_GROUP_LIST
    //     Repeats: 48 times
    /// <summary>
    /// Internal storage, repeating group view count
    /// </summary>
    private int _ExpGroupList_MA;
    /// <summary>
    /// Repeating group view count
    /// </summary>
    public int ExpGroupList_MA {
      get {
        return(_ExpGroupList_MA);
      }
      set {
        _ExpGroupList_MA = value;
      }
    }
    /// <summary>
    /// Internal storage, repeating group view occurrance array
    /// </summary>
    private char[] _ExpGroupList_AC = new char[48];
    /// <summary>
    /// Repeating group view occurrance array
    /// </summary>
    public char[] ExpGroupList_AC {
      get {
        return(_ExpGroupList_AC);
      }
      set {
        _ExpGroupList_AC = value;
      }
    }
    // Entity View: EXP_G_LIST
    //        Type: IYY1_PARENT
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpGListIyy1ParentPinstanceId
    /// </summary>
    private char[] _ExpGListIyy1ParentPinstanceId_AS = new char[48];
    /// <summary>
    /// Attribute missing flag for: ExpGListIyy1ParentPinstanceId
    /// </summary>
    public char[] ExpGListIyy1ParentPinstanceId_AS {
      get {
        return(_ExpGListIyy1ParentPinstanceId_AS);
      }
      set {
        _ExpGListIyy1ParentPinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpGListIyy1ParentPinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string[] _ExpGListIyy1ParentPinstanceId = new string[48];
    /// <summary>
    /// Attribute for: ExpGListIyy1ParentPinstanceId
    /// </summary>
    public string[] ExpGListIyy1ParentPinstanceId {
      get {
        return(_ExpGListIyy1ParentPinstanceId);
      }
      set {
        _ExpGListIyy1ParentPinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpGListIyy1ParentPreferenceId
    /// </summary>
    private char[] _ExpGListIyy1ParentPreferenceId_AS = new char[48];
    /// <summary>
    /// Attribute missing flag for: ExpGListIyy1ParentPreferenceId
    /// </summary>
    public char[] ExpGListIyy1ParentPreferenceId_AS {
      get {
        return(_ExpGListIyy1ParentPreferenceId_AS);
      }
      set {
        _ExpGListIyy1ParentPreferenceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpGListIyy1ParentPreferenceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string[] _ExpGListIyy1ParentPreferenceId = new string[48];
    /// <summary>
    /// Attribute for: ExpGListIyy1ParentPreferenceId
    /// </summary>
    public string[] ExpGListIyy1ParentPreferenceId {
      get {
        return(_ExpGListIyy1ParentPreferenceId);
      }
      set {
        _ExpGListIyy1ParentPreferenceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpGListIyy1ParentPkeyAttrText
    /// </summary>
    private char[] _ExpGListIyy1ParentPkeyAttrText_AS = new char[48];
    /// <summary>
    /// Attribute missing flag for: ExpGListIyy1ParentPkeyAttrText
    /// </summary>
    public char[] ExpGListIyy1ParentPkeyAttrText_AS {
      get {
        return(_ExpGListIyy1ParentPkeyAttrText_AS);
      }
      set {
        _ExpGListIyy1ParentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpGListIyy1ParentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string[] _ExpGListIyy1ParentPkeyAttrText = new string[48];
    /// <summary>
    /// Attribute for: ExpGListIyy1ParentPkeyAttrText
    /// </summary>
    public string[] ExpGListIyy1ParentPkeyAttrText {
      get {
        return(_ExpGListIyy1ParentPkeyAttrText);
      }
      set {
        _ExpGListIyy1ParentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpGListIyy1ParentPsearchAttrText
    /// </summary>
    private char[] _ExpGListIyy1ParentPsearchAttrText_AS = new char[48];
    /// <summary>
    /// Attribute missing flag for: ExpGListIyy1ParentPsearchAttrText
    /// </summary>
    public char[] ExpGListIyy1ParentPsearchAttrText_AS {
      get {
        return(_ExpGListIyy1ParentPsearchAttrText_AS);
      }
      set {
        _ExpGListIyy1ParentPsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpGListIyy1ParentPsearchAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string[] _ExpGListIyy1ParentPsearchAttrText = new string[48];
    /// <summary>
    /// Attribute for: ExpGListIyy1ParentPsearchAttrText
    /// </summary>
    public string[] ExpGListIyy1ParentPsearchAttrText {
      get {
        return(_ExpGListIyy1ParentPsearchAttrText);
      }
      set {
        _ExpGListIyy1ParentPsearchAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpGListIyy1ParentPotherAttrText
    /// </summary>
    private char[] _ExpGListIyy1ParentPotherAttrText_AS = new char[48];
    /// <summary>
    /// Attribute missing flag for: ExpGListIyy1ParentPotherAttrText
    /// </summary>
    public char[] ExpGListIyy1ParentPotherAttrText_AS {
      get {
        return(_ExpGListIyy1ParentPotherAttrText_AS);
      }
      set {
        _ExpGListIyy1ParentPotherAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpGListIyy1ParentPotherAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string[] _ExpGListIyy1ParentPotherAttrText = new string[48];
    /// <summary>
    /// Attribute for: ExpGListIyy1ParentPotherAttrText
    /// </summary>
    public string[] ExpGListIyy1ParentPotherAttrText {
      get {
        return(_ExpGListIyy1ParentPotherAttrText);
      }
      set {
        _ExpGListIyy1ParentPotherAttrText = value;
      }
    }
    // Entity View: EXP_ERROR
    //        Type: IYY1_COMPONENT
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentSeverityCode
    /// </summary>
    private char _ExpErrorIyy1ComponentSeverityCode_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentSeverityCode
    /// </summary>
    public char ExpErrorIyy1ComponentSeverityCode_AS {
      get {
        return(_ExpErrorIyy1ComponentSeverityCode_AS);
      }
      set {
        _ExpErrorIyy1ComponentSeverityCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentSeverityCode
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ExpErrorIyy1ComponentSeverityCode;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentSeverityCode
    /// </summary>
    public string ExpErrorIyy1ComponentSeverityCode {
      get {
        return(_ExpErrorIyy1ComponentSeverityCode);
      }
      set {
        _ExpErrorIyy1ComponentSeverityCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentRollbackIndicator
    /// </summary>
    private char _ExpErrorIyy1ComponentRollbackIndicator_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentRollbackIndicator
    /// </summary>
    public char ExpErrorIyy1ComponentRollbackIndicator_AS {
      get {
        return(_ExpErrorIyy1ComponentRollbackIndicator_AS);
      }
      set {
        _ExpErrorIyy1ComponentRollbackIndicator_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentRollbackIndicator
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ExpErrorIyy1ComponentRollbackIndicator;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentRollbackIndicator
    /// </summary>
    public string ExpErrorIyy1ComponentRollbackIndicator {
      get {
        return(_ExpErrorIyy1ComponentRollbackIndicator);
      }
      set {
        _ExpErrorIyy1ComponentRollbackIndicator = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentOriginServid
    /// </summary>
    private char _ExpErrorIyy1ComponentOriginServid_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentOriginServid
    /// </summary>
    public char ExpErrorIyy1ComponentOriginServid_AS {
      get {
        return(_ExpErrorIyy1ComponentOriginServid_AS);
      }
      set {
        _ExpErrorIyy1ComponentOriginServid_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentOriginServid
    /// Domain: Number
    /// Length: 15
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private double _ExpErrorIyy1ComponentOriginServid;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentOriginServid
    /// </summary>
    public double ExpErrorIyy1ComponentOriginServid {
      get {
        return(_ExpErrorIyy1ComponentOriginServid);
      }
      set {
        _ExpErrorIyy1ComponentOriginServid = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentContextString
    /// </summary>
    private char _ExpErrorIyy1ComponentContextString_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentContextString
    /// </summary>
    public char ExpErrorIyy1ComponentContextString_AS {
      get {
        return(_ExpErrorIyy1ComponentContextString_AS);
      }
      set {
        _ExpErrorIyy1ComponentContextString_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentContextString
    /// Domain: Text
    /// Length: 512
    /// Varying Length: Y
    /// </summary>
    private string _ExpErrorIyy1ComponentContextString;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentContextString
    /// </summary>
    public string ExpErrorIyy1ComponentContextString {
      get {
        return(_ExpErrorIyy1ComponentContextString);
      }
      set {
        _ExpErrorIyy1ComponentContextString = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentReturnCode
    /// </summary>
    private char _ExpErrorIyy1ComponentReturnCode_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentReturnCode
    /// </summary>
    public char ExpErrorIyy1ComponentReturnCode_AS {
      get {
        return(_ExpErrorIyy1ComponentReturnCode_AS);
      }
      set {
        _ExpErrorIyy1ComponentReturnCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentReturnCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ExpErrorIyy1ComponentReturnCode;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentReturnCode
    /// </summary>
    public int ExpErrorIyy1ComponentReturnCode {
      get {
        return(_ExpErrorIyy1ComponentReturnCode);
      }
      set {
        _ExpErrorIyy1ComponentReturnCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentReasonCode
    /// </summary>
    private char _ExpErrorIyy1ComponentReasonCode_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentReasonCode
    /// </summary>
    public char ExpErrorIyy1ComponentReasonCode_AS {
      get {
        return(_ExpErrorIyy1ComponentReasonCode_AS);
      }
      set {
        _ExpErrorIyy1ComponentReasonCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentReasonCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ExpErrorIyy1ComponentReasonCode;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentReasonCode
    /// </summary>
    public int ExpErrorIyy1ComponentReasonCode {
      get {
        return(_ExpErrorIyy1ComponentReasonCode);
      }
      set {
        _ExpErrorIyy1ComponentReasonCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentChecksum
    /// </summary>
    private char _ExpErrorIyy1ComponentChecksum_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentChecksum
    /// </summary>
    public char ExpErrorIyy1ComponentChecksum_AS {
      get {
        return(_ExpErrorIyy1ComponentChecksum_AS);
      }
      set {
        _ExpErrorIyy1ComponentChecksum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentChecksum
    /// Domain: Text
    /// Length: 15
    /// Varying Length: N
    /// </summary>
    private string _ExpErrorIyy1ComponentChecksum;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentChecksum
    /// </summary>
    public string ExpErrorIyy1ComponentChecksum {
      get {
        return(_ExpErrorIyy1ComponentChecksum);
      }
      set {
        _ExpErrorIyy1ComponentChecksum = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public MYY10151_OA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public MYY10151_OA( MYY10151_OA orig )
    {
      ExpGroupList_MA = orig.ExpGroupList_MA;
      Array.Copy( orig._ExpGroupList_AC,
      	ExpGroupList_AC,
      	ExpGroupList_AC.Length );
      Array.Copy( orig._ExpGListIyy1ParentPinstanceId_AS,
      	ExpGListIyy1ParentPinstanceId_AS,
      	ExpGListIyy1ParentPinstanceId_AS.Length );
      Array.Copy( orig._ExpGListIyy1ParentPinstanceId,
      	ExpGListIyy1ParentPinstanceId,
      	ExpGListIyy1ParentPinstanceId.Length );
      Array.Copy( orig._ExpGListIyy1ParentPreferenceId_AS,
      	ExpGListIyy1ParentPreferenceId_AS,
      	ExpGListIyy1ParentPreferenceId_AS.Length );
      Array.Copy( orig._ExpGListIyy1ParentPreferenceId,
      	ExpGListIyy1ParentPreferenceId,
      	ExpGListIyy1ParentPreferenceId.Length );
      Array.Copy( orig._ExpGListIyy1ParentPkeyAttrText_AS,
      	ExpGListIyy1ParentPkeyAttrText_AS,
      	ExpGListIyy1ParentPkeyAttrText_AS.Length );
      Array.Copy( orig._ExpGListIyy1ParentPkeyAttrText,
      	ExpGListIyy1ParentPkeyAttrText,
      	ExpGListIyy1ParentPkeyAttrText.Length );
      Array.Copy( orig._ExpGListIyy1ParentPsearchAttrText_AS,
      	ExpGListIyy1ParentPsearchAttrText_AS,
      	ExpGListIyy1ParentPsearchAttrText_AS.Length );
      Array.Copy( orig._ExpGListIyy1ParentPsearchAttrText,
      	ExpGListIyy1ParentPsearchAttrText,
      	ExpGListIyy1ParentPsearchAttrText.Length );
      Array.Copy( orig._ExpGListIyy1ParentPotherAttrText_AS,
      	ExpGListIyy1ParentPotherAttrText_AS,
      	ExpGListIyy1ParentPotherAttrText_AS.Length );
      Array.Copy( orig._ExpGListIyy1ParentPotherAttrText,
      	ExpGListIyy1ParentPotherAttrText,
      	ExpGListIyy1ParentPotherAttrText.Length );
      ExpErrorIyy1ComponentSeverityCode_AS = orig.ExpErrorIyy1ComponentSeverityCode_AS;
      ExpErrorIyy1ComponentSeverityCode = orig.ExpErrorIyy1ComponentSeverityCode;
      ExpErrorIyy1ComponentRollbackIndicator_AS = orig.ExpErrorIyy1ComponentRollbackIndicator_AS;
      ExpErrorIyy1ComponentRollbackIndicator = orig.ExpErrorIyy1ComponentRollbackIndicator;
      ExpErrorIyy1ComponentOriginServid_AS = orig.ExpErrorIyy1ComponentOriginServid_AS;
      ExpErrorIyy1ComponentOriginServid = orig.ExpErrorIyy1ComponentOriginServid;
      ExpErrorIyy1ComponentContextString_AS = orig.ExpErrorIyy1ComponentContextString_AS;
      ExpErrorIyy1ComponentContextString = orig.ExpErrorIyy1ComponentContextString;
      ExpErrorIyy1ComponentReturnCode_AS = orig.ExpErrorIyy1ComponentReturnCode_AS;
      ExpErrorIyy1ComponentReturnCode = orig.ExpErrorIyy1ComponentReturnCode;
      ExpErrorIyy1ComponentReasonCode_AS = orig.ExpErrorIyy1ComponentReasonCode_AS;
      ExpErrorIyy1ComponentReasonCode = orig.ExpErrorIyy1ComponentReasonCode;
      ExpErrorIyy1ComponentChecksum_AS = orig.ExpErrorIyy1ComponentChecksum_AS;
      ExpErrorIyy1ComponentChecksum = orig.ExpErrorIyy1ComponentChecksum;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static MYY10151_OA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new MYY10151_OA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new MYY10151_OA());
          }
          else 
          {
            MYY10151_OA result = freeArray[--countFree];
            freeArray[countFree] = null;
            result.Reset(  );
            return(result);
          }
        }
      }
    }
    /// <summary>
    /// Static free instance method
    /// </summary>
    
    public void FreeInstance(  )
    {
      lock (freeArray)
      {
        if ( countFree < freeArray.Length )
        {
          freeArray[countFree++] = this;
        }
      }
    }
    /// <summary>
    /// clone constructor
    /// </summary>
    
    public override Object Clone(  )
    {
      return(new MYY10151_OA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ExpGroupList_MA = 0;
      for(int a = 0; a < 48; a++)
      {
        ExpGroupList_AC[ a] = ' ';
        ExpGListIyy1ParentPinstanceId_AS[ a] = ' ';
        ExpGListIyy1ParentPinstanceId[ a] = "00000000000000000000";
        ExpGListIyy1ParentPreferenceId_AS[ a] = ' ';
        ExpGListIyy1ParentPreferenceId[ a] = "00000000000000000000";
        ExpGListIyy1ParentPkeyAttrText_AS[ a] = ' ';
        ExpGListIyy1ParentPkeyAttrText[ a] = "     ";
        ExpGListIyy1ParentPsearchAttrText_AS[ a] = ' ';
        ExpGListIyy1ParentPsearchAttrText[ a] = "                         ";
        ExpGListIyy1ParentPotherAttrText_AS[ a] = ' ';
        ExpGListIyy1ParentPotherAttrText[ a] = "                         ";
      }
      ExpErrorIyy1ComponentSeverityCode_AS = ' ';
      ExpErrorIyy1ComponentSeverityCode = " ";
      ExpErrorIyy1ComponentRollbackIndicator_AS = ' ';
      ExpErrorIyy1ComponentRollbackIndicator = " ";
      ExpErrorIyy1ComponentOriginServid_AS = ' ';
      ExpErrorIyy1ComponentOriginServid = 0.0;
      ExpErrorIyy1ComponentContextString_AS = ' ';
      ExpErrorIyy1ComponentContextString = "";
      ExpErrorIyy1ComponentReturnCode_AS = ' ';
      ExpErrorIyy1ComponentReturnCode = 0;
      ExpErrorIyy1ComponentReasonCode_AS = ' ';
      ExpErrorIyy1ComponentReasonCode = 0;
      ExpErrorIyy1ComponentChecksum_AS = ' ';
      ExpErrorIyy1ComponentChecksum = "               ";
    }
    /// <summary>
    /// Sets the current state of the instance to the VDF version.
    /// </summary>
    public void SetFromVDF( VDF vdf )
    {
      throw new Exception("can only execute SetFromVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Gets the VDF version of the current state of the instance.
    /// </summary>
    public VDF GetVDF(  )
    {
      throw new Exception("can only execute GetVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IExportView orig )
    {
      this.CopyFrom((MYY10151_OA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( MYY10151_OA orig )
    {
      ExpGroupList_MA = orig.ExpGroupList_MA;
      for(int a = 0; a < 48; a++)
      {
        ExpGroupList_AC[ a] = orig.ExpGroupList_AC[ a];
        ExpGListIyy1ParentPinstanceId_AS[ a] = orig.ExpGListIyy1ParentPinstanceId_AS[ a];
        ExpGListIyy1ParentPinstanceId[ a] = orig.ExpGListIyy1ParentPinstanceId[ a];
        ExpGListIyy1ParentPreferenceId_AS[ a] = orig.ExpGListIyy1ParentPreferenceId_AS[ a];
        ExpGListIyy1ParentPreferenceId[ a] = orig.ExpGListIyy1ParentPreferenceId[ a];
        ExpGListIyy1ParentPkeyAttrText_AS[ a] = orig.ExpGListIyy1ParentPkeyAttrText_AS[ a];
        ExpGListIyy1ParentPkeyAttrText[ a] = orig.ExpGListIyy1ParentPkeyAttrText[ a];
        ExpGListIyy1ParentPsearchAttrText_AS[ a] = orig.ExpGListIyy1ParentPsearchAttrText_AS[ a];
        ExpGListIyy1ParentPsearchAttrText[ a] = orig.ExpGListIyy1ParentPsearchAttrText[ a];
        ExpGListIyy1ParentPotherAttrText_AS[ a] = orig.ExpGListIyy1ParentPotherAttrText_AS[ a];
        ExpGListIyy1ParentPotherAttrText[ a] = orig.ExpGListIyy1ParentPotherAttrText[ a];
      }
      ExpErrorIyy1ComponentSeverityCode_AS = orig.ExpErrorIyy1ComponentSeverityCode_AS;
      ExpErrorIyy1ComponentSeverityCode = orig.ExpErrorIyy1ComponentSeverityCode;
      ExpErrorIyy1ComponentRollbackIndicator_AS = orig.ExpErrorIyy1ComponentRollbackIndicator_AS;
      ExpErrorIyy1ComponentRollbackIndicator = orig.ExpErrorIyy1ComponentRollbackIndicator;
      ExpErrorIyy1ComponentOriginServid_AS = orig.ExpErrorIyy1ComponentOriginServid_AS;
      ExpErrorIyy1ComponentOriginServid = orig.ExpErrorIyy1ComponentOriginServid;
      ExpErrorIyy1ComponentContextString_AS = orig.ExpErrorIyy1ComponentContextString_AS;
      ExpErrorIyy1ComponentContextString = orig.ExpErrorIyy1ComponentContextString;
      ExpErrorIyy1ComponentReturnCode_AS = orig.ExpErrorIyy1ComponentReturnCode_AS;
      ExpErrorIyy1ComponentReturnCode = orig.ExpErrorIyy1ComponentReturnCode;
      ExpErrorIyy1ComponentReasonCode_AS = orig.ExpErrorIyy1ComponentReasonCode_AS;
      ExpErrorIyy1ComponentReasonCode = orig.ExpErrorIyy1ComponentReasonCode;
      ExpErrorIyy1ComponentChecksum_AS = orig.ExpErrorIyy1ComponentChecksum_AS;
      ExpErrorIyy1ComponentChecksum = orig.ExpErrorIyy1ComponentChecksum;
    }
  }
}
