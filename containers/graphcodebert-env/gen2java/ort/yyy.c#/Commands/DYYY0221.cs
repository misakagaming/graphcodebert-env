namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: DYYY0221_CHILD_READ              Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:41:05
  //    Target DBMS: ODBC/ADO.NET              User: AliAl
  //    Access Method: <NONE>         
  //
  //    Generation options:
  //    Debug trace option not selected
  //    Data modeling constraint enforcement not selected
  //    Optimized import view initialization not selected
  //    Enforce default values with DBMS not selected
  //    Init unspecified optional fields to NULL not selected
  //
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // using Statements
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  using System;
  using com.ca.gen.vwrt;
  using com.ca.gen.vwrt.types;
  using com.ca.gen.vwrt.vdf;
  using com.ca.gen.csu.exception;
  
  using com.ca.gen.abrt;
  using com.ca.gen.abrt.functions;
  using com.ca.gen.abrt.cascade;
  using com.ca.gen.abrt.manager;
  using com.ca.gen.abrt.trace;
  using com.ca.gen.exits.common;
  using com.ca.gen.odc;
  using System.Data;
  using System.Collections;
  
  public class DYYY0221 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * *
    // ENTITY ACTION STATEMENT STATUS FLAGS 
    // ENTITY VIEW STATUS FLAGS AND         
    // LAST COMMAND FOR EACH ENTITY VIEW    
    // * * * * * * * * * * * * * * * * * * *
    string sl_29360181 = ErrorData.LastStatusNone;
    string Child_es;
    string Child_001cd;
    string Child_lk;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // IMPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    DYYY0221_IA WIa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    DYYY0221_OA WOa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // START OF ENTITY ACTION VIEW
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    /// <summary>
    /// Internal data view storage for: DYYY0221_EA
    /// </summary>
    [Serializable]
    public class DYYY0221_EA : ViewBase, IEntityActionView
    {
      // Entity View: 
      //        Type: CHILD
      /// <summary>
      /// Internal storage for attribute missing flag for: ChildCinstanceId
      /// </summary>
      private char _ChildCinstanceId_AS;
      /// <summary>
      /// Attribute missing flag for: ChildCinstanceId
      /// </summary>
      public char ChildCinstanceId_AS {
        get {
          return(_ChildCinstanceId_AS);
        }
        set {
          _ChildCinstanceId_AS = value;
        }
      }
      /// <summary>
      /// Internal storage, attribute for: ChildCinstanceId
      /// Domain: Timestamp
      /// Length: 20
      /// </summary>
      private string _ChildCinstanceId;
      /// <summary>
      /// Attribute for: ChildCinstanceId
      /// </summary>
      public string ChildCinstanceId {
        get {
          return(_ChildCinstanceId);
        }
        set {
          _ChildCinstanceId = value;
        }
      }
      /// <summary>
      /// Internal storage for attribute missing flag for: ChildCreferenceId
      /// </summary>
      private char _ChildCreferenceId_AS;
      /// <summary>
      /// Attribute missing flag for: ChildCreferenceId
      /// </summary>
      public char ChildCreferenceId_AS {
        get {
          return(_ChildCreferenceId_AS);
        }
        set {
          _ChildCreferenceId_AS = value;
        }
      }
      /// <summary>
      /// Internal storage, attribute for: ChildCreferenceId
      /// Domain: Timestamp
      /// Length: 20
      /// </summary>
      private string _ChildCreferenceId;
      /// <summary>
      /// Attribute for: ChildCreferenceId
      /// </summary>
      public string ChildCreferenceId {
        get {
          return(_ChildCreferenceId);
        }
        set {
          _ChildCreferenceId = value;
        }
      }
      /// <summary>
      /// Internal storage for attribute missing flag for: ChildCcreateUserId
      /// </summary>
      private char _ChildCcreateUserId_AS;
      /// <summary>
      /// Attribute missing flag for: ChildCcreateUserId
      /// </summary>
      public char ChildCcreateUserId_AS {
        get {
          return(_ChildCcreateUserId_AS);
        }
        set {
          _ChildCcreateUserId_AS = value;
        }
      }
      /// <summary>
      /// Internal storage, attribute for: ChildCcreateUserId
      /// Domain: Text
      /// Length: 8
      /// Varying Length: N
      /// </summary>
      private string _ChildCcreateUserId;
      /// <summary>
      /// Attribute for: ChildCcreateUserId
      /// </summary>
      public string ChildCcreateUserId {
        get {
          return(_ChildCcreateUserId);
        }
        set {
          _ChildCcreateUserId = value;
        }
      }
      /// <summary>
      /// Internal storage for attribute missing flag for: ChildCupdateUserId
      /// </summary>
      private char _ChildCupdateUserId_AS;
      /// <summary>
      /// Attribute missing flag for: ChildCupdateUserId
      /// </summary>
      public char ChildCupdateUserId_AS {
        get {
          return(_ChildCupdateUserId_AS);
        }
        set {
          _ChildCupdateUserId_AS = value;
        }
      }
      /// <summary>
      /// Internal storage, attribute for: ChildCupdateUserId
      /// Domain: Text
      /// Length: 8
      /// Varying Length: N
      /// </summary>
      private string _ChildCupdateUserId;
      /// <summary>
      /// Attribute for: ChildCupdateUserId
      /// </summary>
      public string ChildCupdateUserId {
        get {
          return(_ChildCupdateUserId);
        }
        set {
          _ChildCupdateUserId = value;
        }
      }
      /// <summary>
      /// Internal storage for attribute missing flag for: ChildCparentPkeyAttrText
      /// </summary>
      private char _ChildCparentPkeyAttrText_AS;
      /// <summary>
      /// Attribute missing flag for: ChildCparentPkeyAttrText
      /// </summary>
      public char ChildCparentPkeyAttrText_AS {
        get {
          return(_ChildCparentPkeyAttrText_AS);
        }
        set {
          _ChildCparentPkeyAttrText_AS = value;
        }
      }
      /// <summary>
      /// Internal storage, attribute for: ChildCparentPkeyAttrText
      /// Domain: Text
      /// Length: 5
      /// Varying Length: N
      /// </summary>
      private string _ChildCparentPkeyAttrText;
      /// <summary>
      /// Attribute for: ChildCparentPkeyAttrText
      /// </summary>
      public string ChildCparentPkeyAttrText {
        get {
          return(_ChildCparentPkeyAttrText);
        }
        set {
          _ChildCparentPkeyAttrText = value;
        }
      }
      /// <summary>
      /// Internal storage for attribute missing flag for: ChildCkeyAttrNum
      /// </summary>
      private char _ChildCkeyAttrNum_AS;
      /// <summary>
      /// Attribute missing flag for: ChildCkeyAttrNum
      /// </summary>
      public char ChildCkeyAttrNum_AS {
        get {
          return(_ChildCkeyAttrNum_AS);
        }
        set {
          _ChildCkeyAttrNum_AS = value;
        }
      }
      /// <summary>
      /// Internal storage, attribute for: ChildCkeyAttrNum
      /// Domain: Number
      /// Length: 6
      /// Decimal Places: 0
      /// Decimal Precision: N
      /// </summary>
      private int _ChildCkeyAttrNum;
      /// <summary>
      /// Attribute for: ChildCkeyAttrNum
      /// </summary>
      public int ChildCkeyAttrNum {
        get {
          return(_ChildCkeyAttrNum);
        }
        set {
          _ChildCkeyAttrNum = value;
        }
      }
      /// <summary>
      /// Internal storage for attribute missing flag for: ChildCsearchAttrText
      /// </summary>
      private char _ChildCsearchAttrText_AS;
      /// <summary>
      /// Attribute missing flag for: ChildCsearchAttrText
      /// </summary>
      public char ChildCsearchAttrText_AS {
        get {
          return(_ChildCsearchAttrText_AS);
        }
        set {
          _ChildCsearchAttrText_AS = value;
        }
      }
      /// <summary>
      /// Internal storage, attribute for: ChildCsearchAttrText
      /// Domain: Text
      /// Length: 25
      /// Varying Length: N
      /// </summary>
      private string _ChildCsearchAttrText;
      /// <summary>
      /// Attribute for: ChildCsearchAttrText
      /// </summary>
      public string ChildCsearchAttrText {
        get {
          return(_ChildCsearchAttrText);
        }
        set {
          _ChildCsearchAttrText = value;
        }
      }
      /// <summary>
      /// Internal storage for attribute missing flag for: ChildCotherAttrText
      /// </summary>
      private char _ChildCotherAttrText_AS;
      /// <summary>
      /// Attribute missing flag for: ChildCotherAttrText
      /// </summary>
      public char ChildCotherAttrText_AS {
        get {
          return(_ChildCotherAttrText_AS);
        }
        set {
          _ChildCotherAttrText_AS = value;
        }
      }
      /// <summary>
      /// Internal storage, attribute for: ChildCotherAttrText
      /// Domain: Text
      /// Length: 25
      /// Varying Length: N
      /// </summary>
      private string _ChildCotherAttrText;
      /// <summary>
      /// Attribute for: ChildCotherAttrText
      /// </summary>
      public string ChildCotherAttrText {
        get {
          return(_ChildCotherAttrText);
        }
        set {
          _ChildCotherAttrText = value;
        }
      }
      /// <summary>
      /// Default Constructor
      /// </summary>
      
      public DYYY0221_EA(  )
      {
        Reset(  );
      }
      /// <summary>
      /// Copy Constructor
      /// </summary>
      
      public DYYY0221_EA( DYYY0221_EA orig )
      {
        ChildCinstanceId_AS = orig.ChildCinstanceId_AS;
        ChildCinstanceId = orig.ChildCinstanceId;
        ChildCreferenceId_AS = orig.ChildCreferenceId_AS;
        ChildCreferenceId = orig.ChildCreferenceId;
        ChildCcreateUserId_AS = orig.ChildCcreateUserId_AS;
        ChildCcreateUserId = orig.ChildCcreateUserId;
        ChildCupdateUserId_AS = orig.ChildCupdateUserId_AS;
        ChildCupdateUserId = orig.ChildCupdateUserId;
        ChildCparentPkeyAttrText_AS = orig.ChildCparentPkeyAttrText_AS;
        ChildCparentPkeyAttrText = orig.ChildCparentPkeyAttrText;
        ChildCkeyAttrNum_AS = orig.ChildCkeyAttrNum_AS;
        ChildCkeyAttrNum = orig.ChildCkeyAttrNum;
        ChildCsearchAttrText_AS = orig.ChildCsearchAttrText_AS;
        ChildCsearchAttrText = orig.ChildCsearchAttrText;
        ChildCotherAttrText_AS = orig.ChildCotherAttrText_AS;
        ChildCotherAttrText = orig.ChildCotherAttrText;
      }
      /// <summary>
      /// clone constructor
      /// </summary>
      
      public override Object Clone(  )
      {
        return(new DYYY0221_EA(this));
      }
      /// <summary>
      /// Resets all properties to the defaults.
      /// </summary>
      
      public void Reset(  )
      {
        ChildCinstanceId_AS = ' ';
        ChildCinstanceId = "00000000000000000000";
        ChildCreferenceId_AS = ' ';
        ChildCreferenceId = "00000000000000000000";
        ChildCcreateUserId_AS = ' ';
        ChildCcreateUserId = "        ";
        ChildCupdateUserId_AS = ' ';
        ChildCupdateUserId = "        ";
        ChildCparentPkeyAttrText_AS = ' ';
        ChildCparentPkeyAttrText = "     ";
        ChildCkeyAttrNum_AS = ' ';
        ChildCkeyAttrNum = 0;
        ChildCsearchAttrText_AS = ' ';
        ChildCsearchAttrText = "                         ";
        ChildCotherAttrText_AS = ' ';
        ChildCotherAttrText = "                         ";
      }
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ENTITY ACTION VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    DYYY0221_EA WEa = new DYYY0221_EA();
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // LOCAL VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    DYYY0221_LA WLa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // CURSOR OPEN FLAGS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool curs_open_0029360181 = false;
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // HOST VARIABLES FOR TABLE: VDVYYYC
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    int CkeyAttr_001EF;
    int CkeyAttr_002EN;
    string CsearchAttr_003EF;
    string CsearchAttr_004EN;
    string CotherAttr_005EF;
    string CotherAttr_006EN;
    string CcreateUserid_007EF;
    string CcreateUserid_008EN;
    string CupdateUserid_009EF;
    string CupdateUserid_010EN;
    DateTime CinstanceId_011EF;
    DateTime CinstanceId_012EN;
    DateTime CreferenceId_013EF;
    DateTime CreferenceId_014EN;
    string FkVdvyyyppkeyAtt_015EF;
    string FkVdvyyyppkeyAtt_016EN;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // TEMPORARY HOST VARIABLES 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    string CparentPkeyAttrText_001TP;
    int CkeyAttrNum_002TP;
    
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020306_esc_flag;
    IAbstractCommand hstmt_0037932487_1_cmd;
    IAbstractResultSet hstmt_0037932487_1_rs;
    //       +->   DYYY0221_CHILD_READ               01/09/2024  13:41
    //       !       IMPORTS:
    //       !         Work View imp_error iyy1_component (Transient,
    //       !                     Mandatory, Import only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !         Entity View imp child (Transient, Mandatory, Import
    //       !                     only)
    //       !           cinstance_id
    //       !           cparent_pkey_attr_text
    //       !           ckey_attr_num
    //       !       EXPORTS:
    //       !         Entity View exp child (Transient, Export only)
    //       !           cinstance_id
    //       !           creference_id
    //       !           ccreate_user_id
    //       !           cupdate_user_id
    //       !           cparent_pkey_attr_text
    //       !           ckey_attr_num
    //       !           csearch_attr_text
    //       !           cother_attr_text
    //       !         Work View exp_error iyy1_component (Transient, Export
    //       !                     only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !       ENTITY ACTIONS:
    //       !         Entity View child
    //       !           cinstance_id
    //       !           creference_id
    //       !           ccreate_user_id
    //       !           cupdate_user_id
    //       !           cparent_pkey_attr_text
    //       !           ckey_attr_num
    //       !           csearch_attr_text
    //       !           cother_attr_text
    //       !       LOCALS:
    //       !         Work View loc dont_change_return_codes
    //       !           1_ok
    //       !           n10_obj_not_found
    //       !         Work View loc dont_change_reason_codes
    //       !           1_default
    //       !           121_child_not_found
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !  NOTE: 
    //     1 !  See the description for the purpose
    //     1 !  
    //     2 !  NOTE: 
    //     2 !  RELEASE HISTORY
    //     2 !  01_00 23-02-1998 New release
    //     2 !  
    //     3 !  MOVE imp_error iyy1_component TO exp_error iyy1_component
    //     4 !   
    //     5 !  NOTE: 
    //     5 !  ****************************************************************
    //     5 !  The used ReturnCode/ReasonCode values
    //     5 !  
    //     6 !  NOTE: 
    //     6 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     6 !  Set the return and reason codes below
    //     6 !  
    //     7 !  SET loc dont_change_return_codes 1_ok TO 1 
    //     8 !  SET loc dont_change_return_codes n10_obj_not_found TO -10 
    //     9 !   
    //    10 !  SET loc dont_change_reason_codes 1_default TO 1 
    //    11 !  SET loc dont_change_reason_codes 121_child_not_found TO 121 
    //    12 !   
    //    13 !  NOTE: 
    //    13 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    13 !  !!!!!!!!!!!!!!!!!!!!!
    //    13 !  If the Instance_id is being used, reading must be operated
    //    13 !  according to this section, instead 
    //    13 !  of business_key(s).
    //    13 !  
    //    13 !  
    //    13 !  
    //    14 !  NOTE: 
    //    14 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    14 !  !!!!!!!!!!!!!!!!!!!!!
    //    14 !  Choose Control of Cursor Generation property as "Select
    //    14 !  only".
    //    14 !  
    //    15 !  +=>READ (Select Only) child
    //    15 !  !        WHERE DESIRED child cparent_pkey_attr_text = imp
    //    15 !  !              child cparent_pkey_attr_text  
    //    15 !  !              AND  DESIRED child ckey_attr_num = imp child
    //    15 !  !              ckey_attr_num
    //    15 !  +> WHEN successful
    //    16 !  !  MOVE  child TO exp child
    //    15 !  +> WHEN not found
    //    17 !  !  SET exp_error iyy1_component return_code TO loc
    //    17 !  !              dont_change_return_codes n10_obj_not_found 
    //    18 !  !  SET exp_error iyy1_component reason_code TO loc
    //    18 !  !              dont_change_reason_codes 121_child_not_found 
    //    15 !  +--
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public DYYY0221(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:41:05";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "DYYY0221_CHILD_READ";
      NestingLevel = 0;
      ValChkDeadlockTimeout = false;
      ValChkDBError = false;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK FUNCTION DECLARATIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void Execute( Object in_runtime_parm1, 
    	IRuntimePStepContext in_runtime_parm2, 
    	GlobData in_globdata, 
    	DYYY0221_IA import_view, 
    	DYYY0221_OA export_view )
    {
      IefRuntimeParm1 = in_runtime_parm1;
      IefRuntimeParm2 = in_runtime_parm2;
      Globdata = in_globdata;
      WIa = import_view;
      WOa = export_view;
      _Execute();
    }
    
    private void _Execute()
    {
      
      f_22020306_localAlloc( "DYYY0221_CHILD_READ" );
      if ( Globdata.GetErrorData().GetLastStatus() == ErrorData.LastStatusIefAllocationError )
      	return;
      
      ++(NestingLevel);
      try {
        f_22020306_init(  );
        f_22020306(  );
      } catch( Exception e ) {
        if ( ((Globdata.GetErrorData().GetStatus() == ErrorData.StatusNone) && (Globdata.GetErrorData().GetErrorEncountered() == 
          ErrorData.ErrorEncounteredNoErrorFound)) && (Globdata.GetErrorData().GetViewOverflow() == 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
          Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusUnexpectedExceptionError );
          Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
          Globdata.GetErrorData(  ).SetErrorMessage( e );
        }
      }
      --(NestingLevel);
    }
    public void f_22020306(  )
    {
      func_0022020306_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020306" );
      Globdata.GetStateData().SetCurrentABName( "DYYY0221_CHILD_READ" );
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    See the description for the purpose                             
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RELEASE HISTORY                                                 
      //    01_00 23-02-1998 New release                                    
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000003" );
      WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(WIa.ImpErrorIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(WIa.ImpErrorIyy1ComponentRollbackIndicator, 1);
      WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(WIa.ImpErrorIyy1ComponentOriginServid);
      WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(WIa.ImpErrorIyy1ComponentContextString, 512);
      WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(WIa.ImpErrorIyy1ComponentReturnCode);
      WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(WIa.ImpErrorIyy1ComponentReasonCode);
      WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(WIa.ImpErrorIyy1ComponentChecksum, 15);
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    **************************************************************  
      //    **                                                              
      //    The used ReturnCode/ReasonCode values                           
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!                                                              
      //    Set the return and reason codes below                           
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000007" );
      WLa.LocDontChangeReturnCodesQ1Ok = IntAttr.ValueOf((int)TIRD2DEC.Execute1(1, 0, TIRD2DEC.ROUND_NONE, 5));
      Globdata.GetStateData().SetLastStatementNumber( "0000000008" );
      WLa.LocDontChangeReturnCodesN10ObjNotFound = IntAttr.ValueOf((int)TIRD2DEC.Execute1(-10, 0, TIRD2DEC.ROUND_NONE, 5));
      
      Globdata.GetStateData().SetLastStatementNumber( "0000000010" );
      WLa.LocDontChangeReasonCodesQ1Default = IntAttr.ValueOf((int)TIRD2DEC.Execute1(1, 0, TIRD2DEC.ROUND_NONE, 5));
      Globdata.GetStateData().SetLastStatementNumber( "0000000011" );
      WLa.LocDontChangeReasonCodesQ121ChildNotFound = IntAttr.ValueOf((int)TIRD2DEC.Execute1(121, 0, TIRD2DEC.ROUND_NONE, 5));
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!!!!!!!!!!                                           
      //    If the Instance_id is being used, reading must be operated      
      //    according to this section, instead                              
      //    of business_key(s).                                             
      //                                                                    
      //                                                                    
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!!!!!!!!!!                                           
      //    Choose Control of Cursor Generation property as "Select         
      //    only".                                                          
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000015" );
      f_29360181(  );
      if ( sl_29360181 == ErrorData.LastStatusSucceeds )
      {
        Globdata.GetStateData().SetLastSubStatementNumber( "1" );
        {
          Globdata.GetStateData().SetLastStatementNumber( "0000000016" );
          WOa.ExpChildCinstanceId = TimestampAttr.ValueOf(WEa.ChildCinstanceId);
          WOa.ExpChildCreferenceId = TimestampAttr.ValueOf(WEa.ChildCreferenceId);
          WOa.ExpChildCcreateUserId = FixedStringAttr.ValueOf(WEa.ChildCcreateUserId, 8);
          WOa.ExpChildCupdateUserId = FixedStringAttr.ValueOf(WEa.ChildCupdateUserId, 8);
          WOa.ExpChildCparentPkeyAttrText = FixedStringAttr.ValueOf(WEa.ChildCparentPkeyAttrText, 5);
          WOa.ExpChildCkeyAttrNum = IntAttr.ValueOf(WEa.ChildCkeyAttrNum);
          WOa.ExpChildCsearchAttrText = FixedStringAttr.ValueOf(WEa.ChildCsearchAttrText, 25);
          WOa.ExpChildCotherAttrText = FixedStringAttr.ValueOf(WEa.ChildCotherAttrText, 25);
        }
      }
      else if ( sl_29360181 == ErrorData.LastStatusNotFound )
      {
        Globdata.GetStateData().SetLastSubStatementNumber( "2" );
        {
          Globdata.GetStateData().SetLastStatementNumber( "0000000017" );
          WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf((int)TIRD2DEC.Execute1((double) 
            WLa.LocDontChangeReturnCodesN10ObjNotFound, 0, TIRD2DEC.ROUND_NONE, 5));
          Globdata.GetStateData().SetLastStatementNumber( "0000000018" );
          WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf((int)TIRD2DEC.Execute1((double) 
            WLa.LocDontChangeReasonCodesQ121ChildNotFound, 0, TIRD2DEC.ROUND_NONE, 5));
        }
      }
      else {
        Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
        Globdata.GetErrorData().SetLastStatus( sl_29360181 );
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
        throw new ABException();
      }
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020306_localAlloc( String abname )
    {
      // Request localview allocation 
      WLa = (GEN.ORT.YYY.DYYY0221_LA)(IefRuntimeParm2.GetInstance( GetType().Assembly,
      	"GEN.ORT.YYY.DYYY0221_LA" ));
      if ( WLa == null )
      {
        Globdata.GetStateData().SetCurrentABId( "0022020306" );
        Globdata.GetStateData().SetCurrentABName( abname );
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
        Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusIefAllocationError );
      }
    }
    
    public void f_22020306_init(  )
    {
      
      CkeyAttr_001EF = 0;
      CkeyAttr_002EN = 0;
      CsearchAttr_003EF = Spaces;
      CsearchAttr_004EN = Spaces;
      CotherAttr_005EF = Spaces;
      CotherAttr_006EN = Spaces;
      CcreateUserid_007EF = Spaces;
      CcreateUserid_008EN = Spaces;
      CupdateUserid_009EF = Spaces;
      CupdateUserid_010EN = Spaces;
      CinstanceId_011EF = TIRVW2S.Execute( IefRuntimeParm1,
      	IefRuntimeParm2,
      	Globdata,
      	"IEFDB",
      	"00010101000000000000" );
      CinstanceId_012EN = TIRVW2S.Execute( IefRuntimeParm1,
      	IefRuntimeParm2,
      	Globdata,
      	"IEFDB",
      	"00010101000000000000" );
      CreferenceId_013EF = TIRVW2S.Execute( IefRuntimeParm1,
      	IefRuntimeParm2,
      	Globdata,
      	"IEFDB",
      	"00010101000000000000" );
      CreferenceId_014EN = TIRVW2S.Execute( IefRuntimeParm1,
      	IefRuntimeParm2,
      	Globdata,
      	"IEFDB",
      	"00010101000000000000" );
      FkVdvyyyppkeyAtt_015EF = Spaces;
      FkVdvyyyppkeyAtt_016EN = Spaces;
      if ( NestingLevel < 2 )
      {
        WLa.Reset();
      }
      WEa.Reset();
      WOa.ExpChildCinstanceId = "00000000000000000000";
      WOa.ExpChildCreferenceId = "00000000000000000000";
      WOa.ExpChildCcreateUserId = "        ";
      WOa.ExpChildCupdateUserId = "        ";
      WOa.ExpChildCparentPkeyAttrText = "     ";
      WOa.ExpChildCkeyAttrNum = 0;
      WOa.ExpChildCsearchAttrText = "                         ";
      WOa.ExpChildCotherAttrText = "                         ";
      WOa.ExpErrorIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorIyy1ComponentRollbackIndicator = " ";
      WOa.ExpErrorIyy1ComponentOriginServid = 0.0;
      WOa.ExpErrorIyy1ComponentContextString = "";
      WOa.ExpErrorIyy1ComponentReturnCode = 0;
      WOa.ExpErrorIyy1ComponentReasonCode = 0;
      WOa.ExpErrorIyy1ComponentChecksum = "               ";
      Child_es = ABBase.EvUnusable;
      Child_lk = ABBase.EvwUnlocked;
      WEa.ChildCinstanceId = "00000000000000000000";
      WEa.ChildCreferenceId = "00000000000000000000";
    }
    public void f_29360181(  )
    {
      sl_29360181 = ErrorData.LastStatusSucceeds;
      Child_001cd = ABBase.PvSelect;
      Child_lk = ABBase.EvwUnlocked;
      f_29360181_moveb(  );
      
      if ( sl_29360181 == ErrorData.LastStatusSucceeds )
      {
        ValChkDeadlockTimeout = false;
        ValChkDBError = false;
        Child_es = ABBase.EvUsable;
        DataException = null;
        try {
          if ( hstmt_0037932487_1_cmd == null )
          {
            SQLStatement.Length = 0;
            
            SQLStatement.Append("SELECT ");
            SQLStatement.Append("VDVYYYC01.`CINSTANCE_ID`,");
            SQLStatement.Append("VDVYYYC01.`CREFERENCE_ID`,");
            SQLStatement.Append("VDVYYYC01.`CCREATE_USERID`,");
            SQLStatement.Append("VDVYYYC01.`CUPDATE_USERID`,");
            SQLStatement.Append("VDVYYYC01.`FK_VDVYYYPPKEY_ATT`,");
            SQLStatement.Append("VDVYYYC01.`CKEY_ATTR`,");
            SQLStatement.Append("VDVYYYC01.`CSEARCH_ATTR`,");
            SQLStatement.Append("VDVYYYC01.`COTHER_ATTR`");
            SQLStatement.Append(" FROM ");
            SQLStatement.Append("`VDVYYYC` VDVYYYC01");
            SQLStatement.Append(" WHERE ");
            SQLStatement.Append("(");
            SQLStatement.Append("VDVYYYC01.`FK_VDVYYYPPKEY_ATT` = ? AND VDVYYYC01.`CKEY_ATTR` = ?");
            SQLStatement.Append(")");
            hstmt_0037932487_1_cmd = Globdata.GetDBMSData().GetDBMSManager().GetCommand(Globdata, IefRuntimeParm2, "IEFDB", 
              SQLStatement.ToString());
            hstmt_0037932487_1_cmd.InsertParameter();
            hstmt_0037932487_1_cmd.InsertParameter();
          }
          hstmt_0037932487_1_cmd.BindParameter(0, CparentPkeyAttrText_001TP);
          hstmt_0037932487_1_cmd.BindParameter(1, CkeyAttrNum_002TP);
          hstmt_0037932487_1_rs = hstmt_0037932487_1_cmd.ExecuteQuery(  );
          if ( hstmt_0037932487_1_rs.MoveNext(  ) == false )
          {
            throw new Exception("No data found");
          }
          CinstanceId_011EF = Globdata.GetDBMSData().GetDBMSManager().GetDateTime(IefRuntimeParm2, "IEFDB", 
            hstmt_0037932487_1_rs.GetValue(0));
          CreferenceId_013EF = Globdata.GetDBMSData().GetDBMSManager().GetDateTime(IefRuntimeParm2, "IEFDB", 
            hstmt_0037932487_1_rs.GetValue(1));
          CcreateUserid_007EF = Globdata.GetDBMSData().GetDBMSManager().GetString(IefRuntimeParm2, "IEFDB", 
            hstmt_0037932487_1_rs.GetValue(2));
          CupdateUserid_009EF = Globdata.GetDBMSData().GetDBMSManager().GetString(IefRuntimeParm2, "IEFDB", 
            hstmt_0037932487_1_rs.GetValue(3));
          FkVdvyyyppkeyAtt_015EF = Globdata.GetDBMSData().GetDBMSManager().GetString(IefRuntimeParm2, "IEFDB", 
            hstmt_0037932487_1_rs.GetValue(4));
          CkeyAttr_001EF = Globdata.GetDBMSData().GetDBMSManager().GetInt32(IefRuntimeParm2, "IEFDB", hstmt_0037932487_1_rs.GetValue(
            5));
          CsearchAttr_003EF = Globdata.GetDBMSData().GetDBMSManager().GetString(IefRuntimeParm2, "IEFDB", 
            hstmt_0037932487_1_rs.GetValue(6));
          CotherAttr_005EF = Globdata.GetDBMSData().GetDBMSManager().GetString(IefRuntimeParm2, "IEFDB", 
            hstmt_0037932487_1_rs.GetValue(7));
          if ( hstmt_0037932487_1_rs.MoveNext(  ) == true )
          {
            throw new Exception("READ statement with SELECT ONLY property returned more than 1 row");
          }
        } catch( Exception e ) {
          DataException = e;
        }
        if ( Globdata.GetDBMSData().GetDBMSManager().WasSuccessful(DataException) )
        {
          f_29360181_movea(  );
          if ( sl_29360181 == ErrorData.LastStatusSucceeds )
          {
            Child_es = ABBase.EvUsable;
          }
          else {
            Child_es = ABBase.EvUnusable;
          }
          try {
            if ( hstmt_0037932487_1_rs != null )
            {
              hstmt_0037932487_1_rs.Close(  );
              hstmt_0037932487_1_rs = null;
            }
            if ( hstmt_0037932487_1_cmd != null )
            {
              hstmt_0037932487_1_cmd.Close(  );
              hstmt_0037932487_1_cmd = null;
            }
          } catch( Exception e ) {
            if ( Globdata.GetDBMSData().GetDBMSManager().WasSuccessful(e) == false )
            {
              DataException = e;
              f_29360181_adonet_dberror(  );
            }
          }
        }
        else if ( Globdata.GetDBMSData().GetDBMSManager().WasNoDataFound(DataException) )
        {
          sl_29360181 = ErrorData.LastStatusNotFound;
          Child_es = ABBase.EvUnusable;
          try {
            if ( hstmt_0037932487_1_rs != null )
            {
              hstmt_0037932487_1_rs.Close(  );
              hstmt_0037932487_1_rs = null;
            }
            if ( hstmt_0037932487_1_cmd != null )
            {
              hstmt_0037932487_1_cmd.Close(  );
              hstmt_0037932487_1_cmd = null;
            }
          } catch( Exception e ) {
            if ( Globdata.GetDBMSData().GetDBMSManager().WasSuccessful(e) == false )
            {
              DataException = e;
              f_29360181_adonet_dberror(  );
            }
          }
        }
        else {
          f_29360181_adonet_dberror(  );
        }
      }
    }
    
    public void f_29360181_adonet_dberror(  )
    {
      Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
      Globdata.GetDBMSData().SetActionId( 15 );
      if ( DataException != null )
      {
        Globdata.GetErrorData(  ).SetErrorMessage( DataException );
      }
      Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusDbError );
      sl_29360181 = Globdata.GetErrorData().GetLastStatus();
      throw new ABException();
    }
    
    public void f_29360181_moveb(  )
    {
      CparentPkeyAttrText_001TP = WIa.ImpChildCparentPkeyAttrText;
      CkeyAttrNum_002TP = WIa.ImpChildCkeyAttrNum;
    }
    
    public void f_29360181_movea(  )
    {
      WEa.ChildCinstanceId = TimestampAttr.ValueOf(TIRS2VW.Execute( IefRuntimeParm1,
      	IefRuntimeParm2,
      	Globdata,
      	"IEFDB",
      	CinstanceId_011EF ));
      WEa.ChildCreferenceId = TimestampAttr.ValueOf(TIRS2VW.Execute( IefRuntimeParm1,
      	IefRuntimeParm2,
      	Globdata,
      	"IEFDB",
      	CreferenceId_013EF ));
      WEa.ChildCcreateUserId = FixedStringAttr.ValueOf(CcreateUserid_007EF, 8);
      WEa.ChildCupdateUserId = FixedStringAttr.ValueOf(CupdateUserid_009EF, 8);
      WEa.ChildCparentPkeyAttrText = FixedStringAttr.ValueOf(FkVdvyyyppkeyAtt_015EF, 5);
      WEa.ChildCkeyAttrNum = IntAttr.ValueOf(CkeyAttr_001EF);
      WEa.ChildCsearchAttrText = FixedStringAttr.ValueOf(CsearchAttr_003EF, 25);
      WEa.ChildCotherAttrText = FixedStringAttr.ValueOf(CotherAttr_005EF, 25);
    }
  }// end class
  
}

